import torch
import torch.nn.functional as F
import numpy as np
import nvdiffrast.torch as dr


class MeshRenderer:
    """使用 nvdiffrast 渲染 mesh 的类，支持批量渲染和抗锯齿

    参考 pyrender 的实现，使用更标准的相机设置和光照模型
    """

    def __init__(self, image_width=512, image_height=512, device='cuda',
                 fov=16.0, camera_distance=1.0):
        """
        初始化渲染器

        Args:
            image_width: 图像宽度
            image_height: 图像高度
            device: 渲染设备 (cuda/cpu)
            fov: 视场角(度)，默认16度(参考代码中的值)
            camera_distance: 相机距离，默认1.0 (与pyrender保持一致)
        """
        self.image_width = image_width
        self.image_height = image_height
        self.device = device
        self.fov = fov
        self.camera_distance = camera_distance

        # 创建 nvdiffrast 的渲染上下文
        self.glctx = dr.RasterizeCudaContext()

        # 固定相机参数
        self._setup_camera()

    def _setup_camera(self):
        """设置固定的相机 view 和 projection 矩阵

        参考 pyrender 的相机设置:
        - fov: 16度 (相对较小的视场角，适合人脸渲染)
        - near: 0.01, far: 3.0
        - camera_pose: [0, 0, 1] (相机在+Z位置看向原点)
        """
        # 透视投影矩阵 (参考 pyrender)
        fov_y = self.fov / 180.0 * np.pi  # 转换为弧度
        aspect = self.image_width / self.image_height
        near = 0.01
        far = 3.0

        f = 1.0 / np.tan(fov_y / 2.0)
        # OpenGL 透视投影矩阵 (列主序)
        # 用于行向量乘法时需要转置，所以这里直接定义为转置形式
        # 即：v_clip = v_camera @ proj_matrix
        self.proj_matrix = torch.tensor([
            [f/aspect, 0.0, 0.0, 0.0],
            [0.0, f, 0.0, 0.0],
            [0.0, 0.0, (far+near)/(near-far), -1.0],
            [0.0, 0.0, (2*far*near)/(near-far), 0.0]
        ], dtype=torch.float32, device=self.device)

        # 相机位置 (OpenGL convention: camera at +Z looking at origin)
        self.camera_pos = torch.tensor([0.0, 0.0, self.camera_distance],
                                       dtype=torch.float32, device=self.device)

    def _preprocess_vertices(self, verts):
        """
        预处理顶点: 调整到相机空间

        在 OpenGL convention 中:
        - 相机在原点，看向-Z方向
        - 模型需要在相机前方（负Z位置）

        参考 pyrender:
        - camera_pose = [[1,0,0,0], [0,1,0,0], [0,0,1,1], [0,0,0,1]]
        - 相机在 z=1 位置，看向原点 (0,0,0)
        - 模型在原点附近

        Args:
            verts: (N, num_verts, 3) 顶点坐标

        Returns:
            verts_transformed: (N, num_verts, 3) 变换后的坐标
        """
        # FLAME 模型在世界坐标系原点附近
        # 相机在 +Z 位置看向原点，所以模型不需要移动
        # 我们需要做的是将世界坐标转换到相机坐标
        verts_transformed = verts.clone()

        # 相机在 (0, 0, camera_distance) 看向 (0, 0, 0)
        # 将顶点从世界空间转换到相机空间：顶点 - 相机位置
        verts_transformed[:, :, 2] -= self.camera_distance

        return verts_transformed

    def _compute_vertex_normals(self, verts, faces):
        """
        计算顶点法线用于光照

        Args:
            verts: (N, num_verts, 3) 顶点坐标
            faces: (num_faces, 3) 面片索引

        Returns:
            vertex_normals: (N, num_verts, 3) 顶点法线
        """
        batch_size = verts.shape[0]
        num_verts = verts.shape[1]

        # 获取面片的三个顶点
        v0 = verts[:, faces[:, 0], :]  # (N, num_faces, 3)
        v1 = verts[:, faces[:, 1], :]
        v2 = verts[:, faces[:, 2], :]

        # 计算面法线
        e1 = v1 - v0
        e2 = v2 - v0
        face_normals = torch.cross(e1, e2, dim=-1)  # (N, num_faces, 3)
        face_normals = face_normals / \
            (torch.norm(face_normals, dim=-1, keepdim=True) + 1e-6)

        # 将面法线累加到顶点法线（简单平均）
        vertex_normals = torch.zeros(
            batch_size, num_verts, 3, device=self.device)
        for i in range(3):
            idx = faces[:, i].long().unsqueeze(
                0).unsqueeze(-1).expand(batch_size, -1, 3)
            vertex_normals.scatter_add_(1, idx, face_normals)

        # 归一化
        vertex_normals = vertex_normals / \
            (torch.norm(vertex_normals, dim=-1, keepdim=True) + 1e-6)

        return vertex_normals

    def _compute_lighting(self, vertex_normals):
        """
        计算光照 - 精确匹配 pyrender 的多光源设置

        pyrender 使用:
        - 环境光: [0.2, 0.2, 0.2]  
        - 方向光: DirectionalLight, intensity=2
        - 5个光源 (初始位置 + 4个旋转π/6的位置)

        参考 reference.py:
        - light_angle = π/6
        - 初始光源位置: [0, 0, 1]
        - 绕X轴旋转±π/6 (2个)
        - 绕Y轴旋转±π/6 (2个)

        Args:
            vertex_normals: (N, num_verts, 3) 顶点法线

        Returns:
            vertex_colors: (N, num_verts, 3) 带光照的顶点颜色
        """
        # 环境光
        ambient = 0.3

        # 光源强度（降低以避免过曝）
        light_intensity = 0.6

        # pyrender 的初始光源位置 [0, 0, 1]，即从相机位置照射
        light_angle = np.pi / 6.0  # π/6 = 30度

        # 5个光源方向 (模拟 pyrender 的 _get_light_poses)
        # 1. 初始位置
        light_dirs = [
            torch.tensor([0.0, 0.0, 1.0], device=self.device),
        ]

        # 2. 绕X轴旋转 +π/6: Rx(θ) = [[1,0,0], [0,cos,-sin], [0,sin,cos]]
        cos_a = np.cos(light_angle)
        sin_a = np.sin(light_angle)
        light_dirs.append(torch.tensor(
            [0.0, -sin_a, cos_a], device=self.device))  # [0, 0, 1] 绕X轴旋转

        # 3. 绕X轴旋转 -π/6
        light_dirs.append(torch.tensor(
            [0.0, sin_a, cos_a], device=self.device))

        # 4. 绕Y轴旋转 -π/6: Ry(θ) = [[cos,0,sin], [0,1,0], [-sin,0,cos]]
        light_dirs.append(torch.tensor(
            [sin_a, 0.0, cos_a], device=self.device))

        # 5. 绕Y轴旋转 +π/6
        light_dirs.append(torch.tensor(
            [-sin_a, 0.0, cos_a], device=self.device))

        # 归一化所有光源方向
        light_dirs = [F.normalize(ld.reshape(1, 1, 3), dim=-1)
                      for ld in light_dirs]

        # 计算所有光源的漫反射贡献
        diffuse = torch.zeros_like(vertex_normals[..., :1])
        for light_dir in light_dirs:
            diffuse += torch.sum(vertex_normals * light_dir,
                                 dim=-1, keepdim=True).clamp(min=0.0)

        # 平均5个光源的贡献
        diffuse = diffuse / len(light_dirs)

        # 总光照 = 环境光 + 漫反射
        lighting = ambient + diffuse * light_intensity
        lighting = lighting.clamp(min=0.0, max=1.0)

        # 基础颜色 - 调试用，使用白色以便清晰可见
        # 正式发表时会使用 pyrender
        base_color = torch.tensor(
            [1.0, 1.0, 1.0], device=self.device).reshape(1, 1, 3)

        # 应用光照
        vertex_colors = base_color * lighting

        return vertex_colors

    def render_img_list_batchwise(self, verts, faces, batch_size=50):
        batch_size = 50  # 每批渲染50帧
        all_imgs = []

        for start_idx in range(0, verts.shape[0], batch_size):
            end_idx = min(start_idx + batch_size, verts.shape[0])

            batch_verts = verts[start_idx:end_idx]
            batch_imgs = self.render_img_list(
                batch_verts, faces, enable_lighting=True)
            all_imgs.extend(batch_imgs)
        return all_imgs

    def render_img_list(self, verts, faces, enable_lighting=True, background_color=(1.0, 1.0, 1.0)):
        """
        渲染图像列表

        Args:
            verts: (N, num_verts, 3) 顶点坐标张量
            faces: (num_faces, 3) 面片索引张量，所有帧共享相同拓扑
            enable_lighting: 是否启用光照
            background_color: 背景颜色 RGB tuple, 范围 [0, 1]

        Returns:
            img_tensor_list: List[Tensor] 图像张量列表，每个为 (H, W, 3)，值域 [0, 255] uint8
        """
        batch_size = verts.shape[0]

        # 确保数据在正确的设备上
        if not verts.is_cuda:
            verts = verts.to(self.device)
        if not faces.is_cuda:
            faces = faces.to(self.device)

        # 确保 faces 是 int32 类型
        if faces.dtype != torch.int32:
            faces = faces.int()

        # 预处理顶点
        verts_translated = self._preprocess_vertices(verts)

        # 计算法线和光照
        if enable_lighting:
            vertex_normals = self._compute_vertex_normals(
                verts_translated, faces)
            vertex_colors = self._compute_lighting(vertex_normals)
        else:
            # 使用统一颜色
            vertex_colors = torch.ones(
                batch_size, verts.shape[1], 3, device=self.device) * 0.8

        vertex_colors = vertex_colors.contiguous()

        # 转换为齐次坐标
        verts_homo = torch.cat(
            [verts_translated, torch.ones_like(verts_translated[..., :1])], dim=-1)

        # 应用投影变换 (行向量乘法)
        # verts_homo: (N, V, 4), proj_matrix: (4, 4)
        # 结果: (N, V, 4) 裁剪空间坐标
        verts_transformed = torch.matmul(verts_homo, self.proj_matrix)

        # 批量渲染
        rast, rast_db = dr.rasterize(
            self.glctx,
            verts_transformed,
            faces,
            resolution=[self.image_height, self.image_width]
        )

        # 插值颜色
        color, _ = dr.interpolate(vertex_colors, rast, faces)

        # 抗锯齿处理 - 这是关键！
        color = dr.antialias(color, rast, verts_transformed, faces)

        # 提取 alpha 通道（三角形掩码）
        alpha = (rast[..., 3:4] > 0).float()  # (N, H, W, 1)

        # 添加背景
        bg_color = torch.tensor(
            background_color, device=self.device).reshape(1, 1, 1, 3)
        color = color * alpha + bg_color * (1 - alpha)

        # 转换为 uint8 图像
        color = torch.clamp(color * 255.0, 0, 255).to(torch.uint8)

        # 翻转 y 轴（OpenGL 坐标系）
        color = torch.flip(color, [1])

        # 转换为列表
        img_tensor_list = [color[i] for i in range(batch_size)]

        return img_tensor_list
