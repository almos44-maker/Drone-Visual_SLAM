#!/usr/bin/env python3
import os
import glob
import math
import numpy as np
import cv2
from scipy.optimize import least_squares
import pypangolin as pangolin
import OpenGL.GL as gl
import time

#SETTINGS
MIN_TRUST_INLIERS = 30
DT = 1.0 / 30.0
MIN_PARALLAX_PX = 15.0  #min median pixel displacement vs reference frame to estimate a new pose
VEL_DECAY = 0.95        #EKF velocity damping per frame without a measurement (soft zero-velocity)

#loop closure
LC_CHECK_EVERY = 5      #search for a loop every N successful frames
LC_MIN_GAP = 150        #candidate must be at least N frames older than the current frame
LC_MATCH_DIST = 50      #max hamming distance for a "good" descriptor match
LC_MIN_MATCHES = 150    #min good matches to consider a candidate
LC_MIN_INLIERS = 40     #min PnP inliers to accept the loop (geometric verification)


#EKF (Extended Kalman Filter)
#handles smoothing out the drone's position so the tracking doesn't jump around.
#State vector (self.x) tracks 7 things: [x, y, z, yaw, vx, vy, vz]^T.
class DroneEKF:
    def __init__(self):
        self.x = np.zeros((7, 1))
        self.P = np.eye(7) * 0.1
        self.Q = np.eye(7) * 0.01
        self.R_noise = np.eye(4) * 0.05

    def predict(self, dt):
        #calculates predicted position based on current velocity (p_new = p_old + v * dt).
        F = np.eye(7)
        F[0, 4] = dt
        F[1, 5] = dt
        F[2, 6] = dt

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z, dt):
        #Updates prediction with actual measurement from camera.
        #Measurement z = absolute [x, y, z, yaw]; velocity is inferred through F (predict) and P.
        h_x = np.array([
            [self.x[0, 0]],
            [self.x[1, 0]],
            [self.x[2, 0]],
            [self.x[3, 0]]
        ])

        H = np.zeros((4, 7))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0
        H[3, 3] = 1.0

        #K=Kalman Gain
        S = H @ self.P @ H.T + self.R_noise
        K_gain = self.P @ H.T @ np.linalg.inv(S)

        #Find residual (measured vs predicted)
        y_residual = z - h_x
        #Normalize angle
        y_residual[3, 0] = (y_residual[3, 0] + np.pi) % (2 * np.pi) - np.pi

        #Update state and covariance
        self.x = self.x + K_gain @ y_residual
        self.P = (np.eye(7) - K_gain @ H) @ self.P

        return self.x[0, 0], self.x[1, 0], self.x[2, 0], self.x[3, 0]


#Map & Point classes
class Point:
    _id_counter = 0

    def __init__(self, pt3d):
        self.id = Point._id_counter
        Point._id_counter += 1
        self.point = pt3d  # Global 3D coordinate
        self.frames = []   # Frame IDs that see this point

class Map:
    def __init__(self):
        self.points = []
        self.frames = []


#frame management
class Frame:
    _id_counter = 0

    def __init__(self, image):
        self.image = image
        self.id = Frame._id_counter
        Frame._id_counter += 1
        # 4x4 matrix [R | t]
        self.pose = np.eye(4, dtype=np.float64)
        self.keypoints = []
        self.descriptors = None

        self.Timestamp = time.time()
        self.Processed = False
        #Map keypoint index to 3D point object
        self.kp_to_3d = {}  

    @property
    def RotationMatrix(self):
        return self.pose[:3, :3]

    @property
    def TranslationVector(self):
        return self.pose[:3, 3]

    def set_pose(self, R, t):
        self.pose[:3, :3] = R
        self.pose[:3, 3] = t.flatten()
        self.Processed = True


#Visual Odometry pipeline
def calculate_epipolar_error(pts1, pts2, F):
    # Calculates epipolar error using Fundamental matrix constraint: x'^T * F * x = 0
    pts1_h = cv2.convertPointsToHomogeneous(pts1).reshape(-1, 3)
    pts2_h = cv2.convertPointsToHomogeneous(pts2).reshape(-1, 3)
    errors = []
    for p1, p2 in zip(pts1_h, pts2_h):
        line2 = F @ p1
        norm = np.sqrt(line2[0] ** 2 + line2[1] ** 2)
        if norm == 0: continue
        line2 /= norm
        err = np.abs(np.dot(p2, line2))
        errors.append(err)
    return np.mean(errors) if errors else 0.0


class VisualOdometry:
    def __init__(self, K):
        self.K = K.astype(np.float64)
        self.detector = cv2.ORB_create(nfeatures=2500)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.map = Map()
        self.ekf = DroneEKF()
        self.ref_frame = None  # last frame with a successfully estimated pose

        #loop closure
        self.lc_candidates = []     # old frames to search a loop in
        self.lc_count = 0           # counter of successful frames
        self.last_lc_id = -LC_MIN_GAP
        self.loop_edges = []        # (old frame, new frame) pairs for drawing
        self.traj_before_lc = None  # trajectory before the last correction (for drawing)

    def triangulate_points(self, P1, P2, pts1, pts2):
        #SVD to solve A*X = 0 for 3D coordinates from 2D matches
        points_3d = []
        for i in range(len(pts1)):
            A = np.zeros((4, 4))
            u1, v1 = pts1[i]
            u2, v2 = pts2[i]
            A[0] = u1 * P1[2, :] - P1[0, :]
            A[1] = v1 * P1[2, :] - P1[1, :]
            A[2] = u2 * P2[2, :] - P2[0, :]
            A[3] = v2 * P2[2, :] - P2[1, :]
            _, _, Vt = np.linalg.svd(A)
            X = Vt[-1]
            points_3d.append(X[:3] / X[3])
        return np.array(points_3d)

    def reprojection_residuals(self, params, K, pts3d, pts2d):
        #Difference between projected 3D points and original 2D pixels
        theta = params[:3]
        t = params[3:]
        R, _ = cv2.Rodrigues(theta)
        proj_pts2d, _ = cv2.projectPoints(pts3d, R, t, K, distCoeffs=None)
        proj_pts2d = proj_pts2d.reshape(-1, 2)
        return (pts2d - proj_pts2d).flatten()

    def optimize_pose_lm(self, R_initial, t_initial, pts3d, pts2d):
        #Refines camera pose using Levenberg-Marquardt
        theta_initial, _ = cv2.Rodrigues(R_initial)
        initial_params = np.hstack((theta_initial.flatten(), t_initial.flatten()))

        res_before = self.reprojection_residuals(initial_params, self.K, pts3d, pts2d)
        err_before = np.mean(np.abs(res_before))

        result = least_squares(
            self.reprojection_residuals,
            initial_params,
            method='lm',
            args=(self.K, pts3d, pts2d),
            max_nfev=100
        )

        res_after = self.reprojection_residuals(result.x, self.K, pts3d, pts2d)
        err_after = np.mean(np.abs(res_after))

        print(f"   [LM] Reprojection Error -> Before: {err_before:.4f} | After: {err_after:.4f}")

        optimized_R, _ = cv2.Rodrigues(result.x[:3])
        return optimized_R, result.x[3:].reshape(3, 1)

    def detect_loop_closure(self, frame):
        #find the old frame that looks most like the current one (most good descriptor matches)
        best, best_matches = None, []
        for cand in self.lc_candidates:
            if frame.id - cand.id < LC_MIN_GAP:
                break  # candidates are sorted by id, the rest are too recent
            matches = self.matcher.match(cand.descriptors, frame.descriptors)
            good = [m for m in matches if m.distance < LC_MATCH_DIST]
            if len(good) > len(best_matches):
                best, best_matches = cand, good

        if best is None or len(best_matches) < LC_MIN_MATCHES:
            return None

        #geometric verification: PnP against the 3D points of the old frame
        obj_pts = []
        img_pts = []
        for m in best_matches:
            if m.queryIdx in best.kp_to_3d:
                obj_pts.append(best.kp_to_3d[m.queryIdx].point)
                img_pts.append(frame.keypoints[m.trainIdx].pt)

        if len(obj_pts) < LC_MIN_INLIERS:
            return None

        success, rvec, tvec, lc_inliers = cv2.solvePnPRansac(
            np.array(obj_pts, dtype=np.float32),
            np.array(img_pts, dtype=np.float32),
            self.K, None
        )
        if not success or lc_inliers is None or len(lc_inliers) < LC_MIN_INLIERS:
            return None

        R_lc, _ = cv2.Rodrigues(rvec)
        lc_pos = (-R_lc.T @ tvec).flatten()  # current position according to the old map
        return best, lc_pos, len(lc_inliers)

    def apply_loop_correction(self, old_frame, frame, lc_pos):
        #drift = how far the current position is from where the old map says we are
        drift = lc_pos - frame.TranslationVector
        span = frame.id - old_frame.id

        #save trajectory before correction (drawn in red)
        self.traj_before_lc = [f.pose[:3, 3].copy() for f in self.map.frames] + [frame.TranslationVector.copy()]

        #spread the drift linearly over the frames inside the loop (0 at old frame -> full drift now)
        for f in self.map.frames:
            if f.id > old_frame.id:
                f.pose[:3, 3] += drift * (f.id - old_frame.id) / span
        frame.pose[:3, 3] += drift

        #move map points the same way (by the last frame that saw them)
        for p in self.map.points:
            last_id = max(p.frames)
            if last_id > old_frame.id:
                p.point = p.point + drift * (min(last_id, frame.id) - old_frame.id) / span

        self.ekf.x[0:3, 0] = frame.TranslationVector
        self.loop_edges.append((old_frame, frame))
        self.last_lc_id = frame.id
        print(f"   Loop correction | drift: {np.linalg.norm(drift):.2f} over {span} frames")

    def process_frame(self, frame, is_first=False, scale=1.0):
        gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY) if len(frame.image.shape) == 3 else frame.image
        frame.keypoints, frame.descriptors = self.detector.detectAndCompute(gray, None)

        if is_first:
            frame.set_pose(np.eye(3), np.zeros(3))
            self.map.frames.append(frame)
            self.ref_frame = frame
            return

        prev = self.ref_frame
        inliers = 0
        num_matches = 0

        if frame.descriptors is not None and prev.descriptors is not None:
            raw_matches = self.matcher.match(prev.descriptors, frame.descriptors)
            num_matches = len(raw_matches)

            if len(raw_matches) >= 15:
                pts1 = np.float32([prev.keypoints[m.queryIdx].pt for m in raw_matches])
                pts2 = np.float32([frame.keypoints[m.trainIdx].pt for m in raw_matches])

                # Not enough motion vs reference frame- translation would be noise, keep last pose
                parallax = np.median(np.linalg.norm(pts2 - pts1, axis=1))
                if parallax < MIN_PARALLAX_PX:
                    # No measurement -> EKF prediction only (keeps moving with estimated velocity)
                    self.ekf.predict(DT)
                    self.ekf.x[4:7, 0] *= VEL_DECAY
                    frame.pose = self.map.frames[-1].pose.copy()
                    frame.pose[:3, 3] = self.ekf.x[0:3, 0]
                    self.map.frames.append(frame)
                    p = self.ekf.x[0:3, 0]
                    print(f"Frame {frame.id:3} | low parallax ({parallax:.1f}px) vs ref {prev.id} - waiting"
                          f" | pred pos [{p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}]")
                    return

                E, mask = cv2.findEssentialMat(pts1, pts2, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

                if E is not None and E.shape == (3, 3) and mask is not None:
                    # Show matches before / after RANSAC filtering (half size so both fit on screen)
                    inlier_mask = mask.ravel() == 1
                    img_matches_before = cv2.drawMatches(prev.image, prev.keypoints, frame.image, frame.keypoints,
                                                         raw_matches[:100], None, flags=2)
                    good_matches = [m for i, m in enumerate(raw_matches) if inlier_mask[i]]
                    img_matches_after = cv2.drawMatches(prev.image, prev.keypoints, frame.image, frame.keypoints,
                                                        good_matches[:100], None, flags=2)
                    cv2.imshow("1. Matches Before Filter", cv2.resize(img_matches_before, None, fx=0.5, fy=0.5))
                    cv2.imshow("2. Matches After Filter (RANSAC)", cv2.resize(img_matches_after, None, fx=0.5, fy=0.5))
                    cv2.waitKey(1)

                    # Epipolar error: all descriptor matches vs. only RANSAC inliers (after geometry)
                    K_inv = np.linalg.inv(self.K)
                    F_mat = K_inv.T @ E @ K_inv
                    epipolar_err = calculate_epipolar_error(pts1, pts2, F_mat)
                    epipolar_err_after = calculate_epipolar_error(pts1[inlier_mask], pts2[inlier_mask], F_mat)
                    print(f"Frame {frame.id:3} | Epipolar Error -> Before filter: {epipolar_err:.4f} "
                          f"({len(raw_matches)} matches) | After RANSAC: {epipolar_err_after:.4f} "
                          f"({len(good_matches)} inliers)")

                if E is not None and E.shape == (3, 3):
                    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, self.K, mask=mask)
                    inliers = np.sum(mask_pose)

                    if inliers >= MIN_TRUST_INLIERS:
                        P1_local = self.K @ np.hstack((np.eye(3), np.zeros((3, 1))))
                        P2_local = self.K @ np.hstack((R, t))

                        valid_mask = mask_pose.ravel() == 1
                        good_pts1 = pts1[valid_mask]
                        good_pts2 = pts2[valid_mask]
                        valid_indices = [i for i, val in enumerate(valid_mask) if val]

                        pts3d_local = self.triangulate_points(P1_local, P2_local, good_pts1, good_pts2)

                        front_mask = pts3d_local[:, 2] > 0.1
                        pts3d_local = pts3d_local[front_mask]
                        good_pts2_lm = good_pts2[front_mask]
                        final_indices = [valid_indices[i] for i, is_front in enumerate(front_mask) if is_front]

                        if len(pts3d_local) > 10:
                            R_opt, t_opt = self.optimize_pose_lm(R, t, pts3d_local, good_pts2_lm)
                        else:
                            R_opt, t_opt = R, t

                        R_curr = prev.pose[:3, :3] @ R_opt.T
                        t_scaled = t_opt.flatten() * scale
                        t_step_global = prev.pose[:3, :3] @ (-R_opt.T @ t_scaled)

                        abs_yaw = math.atan2(R_curr[1, 0], R_curr[0, 0])
                        t_meas_world = prev.pose[:3, 3] + t_step_global
                        z_meas = np.array([[t_meas_world[0]], [t_meas_world[1]], [t_meas_world[2]], [abs_yaw]])

                        self.ekf.predict(DT)
                        ekf_x, ekf_y, ekf_z, ekf_yaw = self.ekf.update(z_meas, DT)

                        t_ekf_world = np.array([ekf_x, ekf_y, ekf_z])
                        frame.set_pose(R_curr, t_ekf_world)

                        # PnP localization every 10 frames
                        # Runs BEFORE adding this pair's new points, so it localizes against points
                        # triangulated earlier (previous pair) and is not circular.
                        if frame.id % 10 == 0 and frame.id > 0:
                            obj_pts = []
                            img_pts = []
                            for m in raw_matches:
                                if m.queryIdx in prev.kp_to_3d:
                                    obj_pts.append(prev.kp_to_3d[m.queryIdx].point)
                                    img_pts.append(frame.keypoints[m.trainIdx].pt)

                            if len(obj_pts) >= 15:
                                success, rvec, tvec, inliers_pnp = cv2.solvePnPRansac(
                                    np.array(obj_pts, dtype=np.float32),
                                    np.array(img_pts, dtype=np.float32),
                                    self.K, None
                                )
                                if success and inliers_pnp is not None:
                                    R_pnp, _ = cv2.Rodrigues(rvec)
                                    frame.set_pose(R_pnp.T, -R_pnp.T @ tvec)
                                    self.ekf.x[0:3, 0] = frame.TranslationVector.flatten()
                                    print(f"   *** PnP Re-Localization Successful | Inliers: {len(inliers_pnp)} ***")

                        if len(pts3d_local) > 0:
                            pts3d_global = (prev.pose[:3, :3] @ (pts3d_local * scale).T).T + prev.pose[:3, 3]
                            for i, pt in enumerate(pts3d_global):
                                if not np.isnan(pt).any() and np.linalg.norm(pt - prev.pose[:3, 3]) < 200:
                                    new_point = Point(pt)
                                    new_point.frames.extend([prev.id, frame.id])
                                    self.map.points.append(new_point)

                                    m = raw_matches[final_indices[i]]
                                    prev.kp_to_3d[m.queryIdx] = new_point
                                    frame.kp_to_3d[m.trainIdx] = new_point

                        #loop closure detection every LC_CHECK_EVERY successful frames
                        self.lc_count += 1
                        if self.lc_count % LC_CHECK_EVERY == 0 and frame.id - self.last_lc_id > LC_MIN_GAP:
                            result = self.detect_loop_closure(frame)
                            if result is not None:
                                old_frame, lc_pos, n_inliers = result
                                print(f"   <<< LOOP CLOSURE DETECTED >>> with Frame {old_frame.id} | PnP inliers: {n_inliers}")
                                self.apply_loop_correction(old_frame, frame, lc_pos)

                        #keep every 3rd successful frame as a loop candidate (keeps the search fast)
                        if self.lc_count % 3 == 0:
                            self.lc_candidates.append(frame)

                        self.map.frames.append(frame)
                        self.ref_frame = frame
                        print(f"   EKF Pos: [{t_ekf_world[0]:.2f}, {t_ekf_world[1]:.2f}, {t_ekf_world[2]:.2f}]")
                        return

        #No measurement- EKF prediction only
        self.ekf.predict(DT)
        self.ekf.x[4:7, 0] *= VEL_DECAY
        frame.pose = self.map.frames[-1].pose.copy()
        frame.pose[:3, 3] = self.ekf.x[0:3, 0]
        self.map.frames.append(frame)
        if num_matches < 15:
            self.ref_frame = frame  # lost overlap with reference -> restart from current frame
        print(f"Frame {frame.id:3} | TRACKING LOST - FALLBACK (matches: {num_matches}, inliers: {inliers})")


#3D visualization (Pangolin)
class PangolinVisualizer:
    def __init__(self, w=1024, h=768):
        self.quit = False
        pangolin.CreateWindowAndBind('Visual SLAM- 3D Map & Trajectory', w, h)
        gl.glEnable(gl.GL_DEPTH_TEST)

        self.scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(w, h, 420, 420, w // 2, h // 2, 0.1, 1000),
            pangolin.ModelViewLookAt(-10, 10, 10, 0, 0, 0, pangolin.AxisY))  # behind-left-above start (forward= -Z in GL)
        self.handler = pangolin.Handler3D(self.scam)
        self.dcam = pangolin.CreateDisplay().SetBounds(
            pangolin.Attach(0.0), pangolin.Attach(1.0),
            pangolin.Attach(0.0), pangolin.Attach(1.0), -float(w) / h).SetHandler(self.handler)
        self.texture = None

    def render(self, vo, img, keypoints):
        if pangolin.ShouldQuit():
            self.quit = True
            return

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(0.05, 0.05, 0.05, 1)

        #Show 2D feed
        vis_img = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0))
        rgb_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
        h_img, w_img = rgb_img.shape[:2]
        if self.texture is None:
            self.texture = pangolin.GlTexture(w_img, h_img, gl.GL_RGB, True, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        self.texture.Upload(cv2.flip(rgb_img, 0), gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        gl.glViewport(0, int(768 * 0.7), int(1024 * 0.3), int(768 * 0.3))
        gl.glColor3f(1, 1, 1)  # reset color, otherwise the texture by the last glColor (blue Z axis)
        self.texture.RenderToViewport()

        gl.glViewport(0, 0, 1024, 768)
        self.dcam.Activate(self.scam)

        #Draw grid
        grid_limit = 75
        grid_step = 5
        gl.glColor3f(0.2, 0.2, 0.2)
        gl.glBegin(gl.GL_LINES)
        for i in range(-grid_limit, grid_limit + 1, grid_step):
            gl.glVertex3f(i, 0, -grid_limit); gl.glVertex3f(i, 0, grid_limit)
            gl.glVertex3f(-grid_limit, 0, i); gl.glVertex3f(grid_limit, 0, i)
        gl.glEnd()

        # OpenCV camera frame: x right, y DOWN, z forward. OpenGL: x right, y UP, z toward viewer.
        # Drawing (x, -y, -z) = 180 deg rotation about X: "up" is drawn up and the scene is not mirrored.

        #Draw point cloud
        if len(vo.map.points) > 0:
            gl.glPointSize(1)
            gl.glColor3f(0.8, 0.8, 0.8)
            gl.glBegin(gl.GL_POINTS)
            for p in vo.map.points:
                gl.glVertex3f(p.point[0], -p.point[1], -p.point[2])
            gl.glEnd()

        #Draw trajectory
        if len(vo.map.frames) > 1:
            gl.glDisable(gl.GL_DEPTH_TEST)
            gl.glColor3f(0, 1, 0)
            gl.glLineWidth(3)
            gl.glBegin(gl.GL_LINE_STRIP)
            for f in vo.map.frames:
                gl.glVertex3f(f.pose[0, 3], -f.pose[1, 3], -f.pose[2, 3])
            gl.glEnd()
            gl.glEnable(gl.GL_DEPTH_TEST)

        #Draw trajectory before loop closure correction (red) + loop links (yellow)
        if vo.traj_before_lc is not None:
            gl.glDisable(gl.GL_DEPTH_TEST)
            gl.glColor3f(1, 0, 0)
            gl.glLineWidth(1)
            gl.glBegin(gl.GL_LINE_STRIP)
            for p in vo.traj_before_lc:
                gl.glVertex3f(p[0], -p[1], -p[2])
            gl.glEnd()

            gl.glColor3f(1, 1, 0)
            gl.glLineWidth(2)
            gl.glBegin(gl.GL_LINES)
            for f_old, f_new in vo.loop_edges:
                gl.glVertex3f(f_old.pose[0, 3], -f_old.pose[1, 3], -f_old.pose[2, 3])
                gl.glVertex3f(f_new.pose[0, 3], -f_new.pose[1, 3], -f_new.pose[2, 3])
            gl.glEnd()
            gl.glEnable(gl.GL_DEPTH_TEST)

        #draw current camera axes (in camera frame: X right=red, Y down=green, Z forward=blue)
        if len(vo.map.frames) > 0:
            gl.glDisable(gl.GL_DEPTH_TEST)
            gl.glPushMatrix()
            cv_to_gl = np.diag([1.0, -1.0, -1.0, 1.0])
            gl.glMultMatrixd((cv_to_gl @ vo.map.frames[-1].pose).T.flatten())
            gl.glLineWidth(3)
            gl.glBegin(gl.GL_LINES)
            gl.glColor3f(1, 0, 0); gl.glVertex3f(0, 0, 0); gl.glVertex3f(1, 0, 0) # X
            gl.glColor3f(0, 1, 0); gl.glVertex3f(0, 0, 0); gl.glVertex3f(0, 1, 0) # Y
            gl.glColor3f(0, 0, 1); gl.glVertex3f(0, 0, 0); gl.glVertex3f(0, 0, 1) # Z
            gl.glEnd()
            gl.glPopMatrix()
            gl.glEnable(gl.GL_DEPTH_TEST)

        pangolin.FinishFrame()


#main execution loop
def main():
    dataset = "Dataset_VO"
    imgs = sorted(glob.glob(os.path.join(dataset, "*.png")))
    if not imgs:
        print(f"Error: No images found in {dataset}")
        return

    first_img = cv2.imread(imgs[0])
    h, w = first_img.shape[:2]
    K = np.array([[w, 0, w / 2],
                  [0, w, h / 2],
                  [0, 0, 1]], dtype=np.float64)

    vo = VisualOdometry(K)
    vis = PangolinVisualizer()

    print("Starting Visual SLAM")
    for i, p in enumerate(imgs):
        if vis.quit: break
        img = cv2.imread(p)
        frame = Frame(img)
        vo.process_frame(frame, is_first=(i == 0), scale=5.0)
        vis.render(vo, img, frame.keypoints)
        time.sleep(0.03)  # 30 frames per second pace

    print("Processing complete. Visualizer remains open.")
    while not pangolin.ShouldQuit() and not vis.quit:
        vis.render(vo, img, frame.keypoints)


if __name__ == "__main__":
    main()