# ros2 service call /get_best_pose custom_msg_srv/srv/GetBestPose "{is_left_hand: 1}"
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.time import Time

from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import Header
from cv_bridge import CvBridge
from ultralytics import YOLO
import numpy as np
import cv2
import math
import threading
import os

# 3D点云和姿态计算的核心依赖
import open3d as o3d
from scipy.spatial.transform import Rotation

from tf2_ros import TransformBroadcaster

# 导入您的 Service 类型
from custom_msg_srv.srv import GetBestPose

# 用于点云转换的ROS库
from sensor_msgs_py import point_cloud2

class YoloPoseService(Node):
    def __init__(self):
        super().__init__('giim_pose_service_node_3d')

        # --- 缓存最新消息的变量和线程锁 ---
        self._latest_rgb_msg = None
        self._latest_depth_msg = None
        self._lock = threading.Lock()

        self.max_time_diff_ns = 0.05 * 1e9 

        # --- 初始化相机、模型等 ---
        self.camera_intrinsics_matrix = None
        self.o3d_intrinsics = None
        self.intrinsics_set = False

        try:
            # yolo_model_path = "/home/yake/ros2_ws/src/giim_seg/best.pt"
            current_file_path = os.path.abspath(__file__)

            # 2. 从文件路径中获取该文件所在的目录
            #    Get the directory containing the file from the file path
            current_dir = os.path.dirname(current_file_path)

            # 3. 定义模型文件的名称
            #    Define the model file name
            model_filename = "best.pt"

            # 4. 使用 os.path.join() 安全地将目录和文件名拼接为相对路径
            #    Use os.path.join() to safely combine the directory and file name into a relative path
            #    这会自动处理不同操作系统下的路径分隔符（例如 / 或 \）
            #    (This automatically handles path separators for different OSes, e.g., / or \)
            yolo_model_path = os.path.join(current_dir, model_filename)

            # 打印结果以确认
            # Print the results for confirmation
            print(f"Model load Path: {yolo_model_path}")

            self.segmentation_model = YOLO(yolo_model_path)
            self.get_logger().info(f"YOLO model loaded successfully from {yolo_model_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to load YOLO model: {e}")
            return

        self.bridge = CvBridge()
        self.tf_broadcaster = TransformBroadcaster(self)

        # === 调试参数 ===
        self.declare_parameter('debug_publish_pcd', True)
        self.declare_parameter('debug_save_pcd', False)

        # (ROI 和 RANSAC 参数保持不变)
        self.declare_parameter('roi.x_min', -1.0)
        self.declare_parameter('roi.x_max', 1.0)
        self.declare_parameter('roi.y_min', -0.5)
        self.declare_parameter('roi.y_max', 0.4)
        self.declare_parameter('roi.z_min', -0.2)
        self.declare_parameter('roi.z_max', 5.0)
        self.declare_parameter('ransac.table_angle_tolerance_deg', 30.0)
        self.declare_parameter('ransac.table_search_radius', 0.4)

        # === 回调组 ===
        self.service_group = ReentrantCallbackGroup()
        self.subscription_group = ReentrantCallbackGroup()

        # --- ROS 通信接口 ---
        self.pose_image_pub = self.create_publisher(Image, "/giim_pose/result_image", 10)
        self.pub_pc_scene = self.create_publisher(PointCloud2, "/giim_pose/debug/scene", 10)
        self.pub_pc_object = self.create_publisher(PointCloud2, "/giim_pose/debug/object", 10)
        self.pub_pc_table = self.create_publisher(PointCloud2, "/giim_pose/debug/table", 10)
        self.pub_pc_top = self.create_publisher(PointCloud2, "/giim_pose/debug/top_surface", 10)
        
        self.srv = self.create_service(
            GetBestPose, 'get_best_pose', 
            self._service_callback,
            callback_group=self.service_group)
        
        # (订阅者部分)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, "/camera/camera/aligned_depth_to_color/camera_info",
            self.camera_info_callback, 10,
            callback_group=self.subscription_group)
        self.rgb_sub = self.create_subscription(
            Image, "/camera/camera/color/image_raw", 
            self._rgb_callback, 1,
            callback_group=self.subscription_group)
        self.depth_sub = self.create_subscription(
            Image, "/camera/camera/aligned_depth_to_color/image_raw", 
            self._depth_callback, 1,
            callback_group=self.subscription_group)
        
        self.get_logger().info("YOLO 3D Pose Service (V14 - Debug Fix) is ready.")

    # ( _o3d_to_ros_pcd, _rgb_callback, _depth_callback, camera_info_callback 保持不变 )
    def _o3d_to_ros_pcd(self, o3d_pc, header):
        fields = [PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)]
        points_xyz = np.asarray(o3d_pc.points, dtype=np.float32)
        return point_cloud2.create_cloud(header, fields, points_xyz)
    def _rgb_callback(self, msg):
        with self._lock: self._latest_rgb_msg = msg
    def _depth_callback(self, msg):
        with self._lock: self._latest_depth_msg = msg
    def camera_info_callback(self, msg):
        if self.intrinsics_set: return
        try:
            self.camera_intrinsics_matrix = np.array(msg.k).reshape(3, 3)
            self.o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
                msg.width, msg.height, msg.k[0], msg.k[4], msg.k[2], msg.k[5])
            self.intrinsics_set = True
            self.destroy_subscription(self.camera_info_sub)
            self.get_logger().info(f"Camera intrinsics loaded and set for Open3D.")
        except Exception as e:
            self.get_logger().error(f"Failed to parse CameraInfo: {e}")

    def _service_callback(self, request, response):
        """
        V14 (Debug Fix):
        - 修复了 V13 中循环逻辑导致调试点云无法发布的bug
        """
        self.get_logger().info(f"Received pose request: is_left_hand = {request.is_left_hand}")

        with self._lock:
            rgb_msg = self._latest_rgb_msg
            depth_msg = self._latest_depth_msg

        if rgb_msg is None or depth_msg is None:
            response.success = False; response.message = "No image data received."
            return response
        if not self.intrinsics_set:
            response.success = False; response.message = "Camera intrinsics not set."
            return response
        if abs(Time.from_msg(rgb_msg.header.stamp).nanoseconds - 
               Time.from_msg(depth_msg.header.stamp).nanoseconds) > self.max_time_diff_ns:
            response.success = False; response.message = "Data is not synchronized."
            return response
        
        debug_save = self.get_parameter('debug_save_pcd').get_parameter_value().bool_value
        debug_publish = self.get_parameter('debug_publish_pcd').get_parameter_value().bool_value
            
        try:
            # --- 第1步：数据准备与YOLO检测 ---
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            depth_image_raw = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
            depth_image = depth_image_raw.astype(np.uint16)
            results = self.segmentation_model(source=rgb_image, verbose=False)
            result = results[0]

            if result.masks is None or len(result.masks) == 0:
                response.success = False; response.message = "No objects detected."
                return response

            num_objects = len(result.boxes)
            if num_objects == 0:
                response.success = False; response.message = "No objects detected."
                return response
            
            # --- 第2步：准备全局点云 (用于桌面检测) ---
            o3d_rgb = o3d.geometry.Image(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
            o3d_depth = o3d.geometry.Image(depth_image)
            target_height, target_width = depth_image.shape
            
            scene_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d_rgb, o3d_depth, depth_scale=1000.0, depth_trunc=5.0, convert_rgb_to_intensity=False)
            PC_Scene_Raw = o3d.geometry.PointCloud.create_from_rgbd_image(scene_rgbd, self.o3d_intrinsics)
            PC_Scene_Raw.remove_non_finite_points()
            
            # (ROI 裁剪)
            min_x = self.get_parameter('roi.x_min').get_parameter_value().double_value
            max_x = self.get_parameter('roi.x_max').get_parameter_value().double_value
            min_y = self.get_parameter('roi.y_min').get_parameter_value().double_value
            max_y = self.get_parameter('roi.y_max').get_parameter_value().double_value
            min_z = self.get_parameter('roi.z_min').get_parameter_value().double_value
            max_z = self.get_parameter('roi.z_max').get_parameter_value().double_value
            min_bound = np.array([min_x, min_y, min_z]); max_bound = np.array([max_x, max_y, max_z])
            roi_box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
            PC_Scene = PC_Scene_Raw.crop(roi_box)
            if not PC_Scene.has_points():
                response.success = False; response.message = "No points left in Scene after ROI cropping."
                return response
            if debug_publish: self.pub_pc_scene.publish(self._o3d_to_ros_pcd(PC_Scene, rgb_msg.header))

            # (法线过滤)
            PC_Scene_Down = PC_Scene.voxel_down_sample(voxel_size=0.01)
            PC_Scene_Down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            table_angle_tolerance_deg = self.get_parameter('ransac.table_angle_tolerance_deg').get_parameter_value().double_value
            angle_tolerance = np.deg2rad(table_angle_tolerance_deg)
            expected_normal = np.array([0, 0, -1]) 
            scene_normals = np.asarray(PC_Scene_Down.normals)
            dot_products = np.dot(scene_normals, expected_normal)
            horizontal_indices = np.where(np.abs(dot_products) > np.cos(angle_tolerance))[0]
            if len(horizontal_indices) == 0:
                response.success = False; response.message = "No horizontal surfaces (Z-axis) found in ROI."
                return response
            PC_Horizontal_Surfaces = PC_Scene_Down.select_by_index(horizontal_indices)
            
            # === 第3步: 遍历所有目标，计算它们的3D位置和调试数据 ===
            
            valid_targets = [] # 存储 (x_coord, index, position_vec, N_Top_vec, PC_Top_Surface_obj, PC_Object_obj, PC_Table_obj)

            for i in range(num_objects):
                # --- [循环内] 提取第 i 个物体的点云 ---
                mask_proto = result.masks.data[i].cpu().numpy()
                mask_full_size = cv2.resize(mask_proto, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
                mask_bool = (mask_full_size > 0)
                masked_depth_image = depth_image.copy(); masked_depth_image[mask_bool == False] = 0
                o3d_masked_depth = o3d.geometry.Image(masked_depth_image)
                obj_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d_rgb, o3d_masked_depth, depth_scale=1000.0, depth_trunc=5.0, convert_rgb_to_intensity=False)
                PC_Object = o3d.geometry.PointCloud.create_from_rgbd_image(obj_rgbd, self.o3d_intrinsics)
                PC_Object.remove_non_finite_points()
                
                if not PC_Object.has_points():
                    self.get_logger().warn(f"Object {i} has no valid depth. Skipping.")
                    continue

                # --- [循环内] 在物体附近搜索桌面 ---
                obj_centroid = PC_Object.get_center()
                search_radius = self.get_parameter('ransac.table_search_radius').get_parameter_value().double_value
                search_radius_sq = search_radius * search_radius
                horizontal_points = np.asarray(PC_Horizontal_Surfaces.points)
                distances_sq = np.sum(np.square(horizontal_points[:, [0, 2]] - obj_centroid[[0, 2]]), axis=1)
                nearby_indices = np.where(distances_sq < search_radius_sq)[0]
                
                if len(nearby_indices) == 0:
                    self.get_logger().warn(f"Object {i} has no nearby table points. Skipping.")
                    continue
                
                PC_Table_Candidates = PC_Horizontal_Surfaces.select_by_index(nearby_indices)
                table_plane_model, table_inliers = PC_Table_Candidates.segment_plane(
                    distance_threshold=0.01, ransac_n=3, num_iterations=1000)
                PC_Table = PC_Table_Candidates.select_by_index(table_inliers) # <-- 这是桌子
                N_Table = table_plane_model[:3];
                if N_Table[2] > 0: N_Table = -N_Table

                # --- [循环内] 提取顶面并计算位置 ---
                PC_Top_Surface = None
                N_Top = None
                try:
                    PC_Object.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
                    obj_normals = np.asarray(PC_Object.normals)
                    obj_dot_products = np.dot(obj_normals, N_Table)
                    top_indices = np.where(np.abs(obj_dot_products) > np.cos(np.deg2rad(20)))[0]
                    if len(top_indices) > 0:
                        PC_Top_Candidates = PC_Object.select_by_index(top_indices)
                        top_plane_model, top_inliers = PC_Top_Candidates.segment_plane(
                            distance_threshold=0.005, ransac_n=3, num_iterations=500)
                        PC_Top_Surface = PC_Top_Candidates.select_by_index(top_inliers)
                        N_Top = top_plane_model[:3]
                except Exception: pass

                if PC_Top_Surface is None or len(PC_Top_Surface.points) < 20:
                    table_d = table_plane_model[3]
                    obj_points = np.asarray(PC_Object.points)
                    distances_to_table = np.dot(obj_points, N_Table) + table_d
                    max_height_dist = np.min(distances_to_table)
                    top_indices = np.where(distances_to_table < max_height_dist + 0.005)[0]
                    PC_Top_Surface = PC_Object.select_by_index(top_indices)
                    N_Top = N_Table
                
                if not PC_Top_Surface.has_points():
                    self.get_logger().warn(f"Object {i} could not isolate top surface. Skipping.")
                    continue
                
                obj_position, _ = PC_Top_Surface.compute_mean_and_covariance()
                
                # === 核心修改: 存储所有调试点云 ===
                valid_targets.append((
                    obj_position[0],  # X coord for sorting
                    i,                # Index
                    obj_position,     # Final position
                    N_Top,            # Final normal
                    PC_Top_Surface,   # Final top surface cloud
                    PC_Object,        # DEBUG: Object cloud
                    PC_Table          # DEBUG: Table cloud
                ))
            # --- 循环结束 ---

            # === 第4步: 根据请求筛选目标 ===
            if not valid_targets:
                response.success = False; response.message = "No valid targets found after 3D processing."
                return response

            if request.is_left_hand:
                self.get_logger().info(f"Requested Left Hand: Selecting target with MIN X coordinate.")
                target_tuple = min(valid_targets, key=lambda item: item[0])
            else:
                self.get_logger().info(f"Requested Right Hand (default): Selecting target with MAX X coordinate.")
                target_tuple = max(valid_targets, key=lambda item: item[0])

            # 解包我们选中的目标
            best_idx = target_tuple[1]
            final_position = target_tuple[2]
            N_Top = target_tuple[3]
            PC_Top_Surface = target_tuple[4]
            PC_Object_best = target_tuple[5] # <-- 解包调试点云
            PC_Table_best = target_tuple[6]  # <-- 解包调试点云
            
            final_X, final_Y, final_Z = final_position
            
            # === 第5步: 仅为选中的目标计算姿态和发布调试PCD ===
            
            # [调试发布]
            if debug_save:
                o3d.io.write_point_cloud("debug_object_best.pcd", PC_Object_best)
                o3d.io.write_point_cloud("debug_table_best.pcd", PC_Table_best)
                o3d.io.write_point_cloud("debug_top_surface_best.pcd", PC_Top_Surface)
            if debug_publish:
                self.pub_pc_object.publish(self._o3d_to_ros_pcd(PC_Object_best, rgb_msg.header))
                self.pub_pc_table.publish(self._o3d_to_ros_pcd(PC_Table_best, rgb_msg.header))
                self.pub_pc_top.publish(self._o3d_to_ros_pcd(PC_Top_Surface, rgb_msg.header))
            
            # (姿态计算)
            _, cov_matrix = PC_Top_Surface.compute_mean_and_covariance()
            if N_Top[2] > 0: N_Top = -N_Top
            Z_axis = -N_Top 
            Z_axis = Z_axis / np.linalg.norm(Z_axis)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            X_vec_candidate = eigenvectors[:, np.argmax(eigenvalues)]
            X_axis = X_vec_candidate - np.dot(X_vec_candidate, Z_axis) * Z_axis
            X_axis_norm = np.linalg.norm(X_axis)
            if X_axis_norm < 1e-6:
                arbitrary_X = np.array([1.0, 0.0, 0.0])
                X_axis = arbitrary_X - np.dot(arbitrary_X, Z_axis) * Z_axis
                X_axis_norm = np.linalg.norm(X_axis)
            X_axis = X_axis / X_axis_norm
            Y_axis = np.cross(Z_axis, X_axis)
            rot_matrix = np.array([X_axis, Y_axis, Z_axis]).T 
            final_orientation = Rotation.from_matrix(rot_matrix).as_quat()
            final_x, final_y, final_z, final_w = final_orientation
            
            # (6. 发布与响应)
            class_name = result.names[int(result.boxes[best_idx].cls.cpu().numpy()[0])]
            
            t = TransformStamped()
            t.header.stamp = rgb_msg.header.stamp; t.header.frame_id = rgb_msg.header.frame_id
            t.child_frame_id = f"target_{class_name}_3D_best"
            
            t.transform.translation.x = final_X; t.transform.translation.y = final_Y; t.transform.translation.z = final_Z
            t.transform.rotation.x = final_x; t.transform.rotation.y = final_y; t.transform.rotation.z = final_z; t.transform.rotation.w = final_w
            self.tf_broadcaster.sendTransform(t)

            # (可视化)
            annotated_img = rgb_image.copy()
            box = result.boxes[best_idx]
            xyxy = box.xyxy.cpu().numpy()[0].astype(int)
            mask_contour = result.masks.xy[best_idx].astype(np.int32)
            cv2.drawContours(annotated_img, [mask_contour], -1, (255, 255, 0), 2) # Cyan
            conf = float(box.conf.cpu().numpy()[0])
            label_text = f"{class_name} {conf:.2f}"
            cv2.putText(annotated_img, label_text, (xyxy[0], xyxy[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            K = self.camera_intrinsics_matrix
            proj_point_3d = (K @ final_position).flatten()
            u = int(proj_point_3d[0] / proj_point_3d[2]); v = int(proj_point_3d[1] / proj_point_3d[2])
            cv2.drawMarker(annotated_img, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 40, 4)
            
            img_msg = self.bridge.cv2_to_imgmsg(annotated_img, "bgr8"); img_msg.header = rgb_msg.header
            self.pose_image_pub.publish(img_msg)

            # (填充响应)
            response.success = True; response.message = f"Successfully found 3D pose for: {class_name}"
            response.pose.header = rgb_msg.header
            response.pose.pose.position.x = final_X; response.pose.pose.position.y = final_Y; response.pose.pose.position.z = final_Z
            response.pose.pose.orientation.x = final_x; response.pose.pose.orientation.y = final_y; response.pose.pose.orientation.z = final_z; response.pose.pose.orientation.w = final_w
            
            return response

        except Exception as e:
            self.get_logger().error(f"An error occurred during 3D processing: {e}", exc_info=True)
            response.success = False
            response.message = str(e)
            return response


# === 启动多线程执行器 (不变) ===
def main(args=None):
    rclpy.init(args=args)
    yolo_pose_service_node = YoloPoseService()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(yolo_pose_service_node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        yolo_pose_service_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()