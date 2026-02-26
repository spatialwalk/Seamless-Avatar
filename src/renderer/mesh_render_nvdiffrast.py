import numpy as np
import torch
import nvdiffrast.torch as dr
import imageio
import tqdm


class MeshRendererNvdiffrast:
    def __init__(self, fig_resolution,bg_color = [1.0, 1.0, 1.0]):
        """
        初始化 nvdiffrast 渲染器
        
        Args:
            fig_resolution: (height, width) 分辨率
        """
        self.resolution = fig_resolution  # (height, width)
        self.background_color = torch.tensor(bg_color, device='cuda')  # 白色背景 (R, G, B)
        self.glctx = dr.RasterizeCudaContext()
        
    def render_to_img(self, vertices, faces, color, camera, light_dir, save_img_path=None):
        """
        渲染单帧图像
        
        Args:
            vertices: (N, 3) numpy array，世界坐标系中的顶点
            faces: (F, 3) numpy array，面索引
            color: [R, G, B, A] 统一颜色（0-255）
            camera: (4, 4) numpy array，相机变换矩阵
            light_dir: [x, y, z] 光照方向
            save_img_path: 可选，保存路径
        
        Returns:
            img: (H, W, 3) numpy array，渲染的图像（0-255 uint8）
        """
        # 添加 batch 维度并调用 render_to_video
        vertices_batch = vertices[np.newaxis, ...]  # (1, N, 3)
        img_list = self.render_to_video(vertices_batch, faces, color, camera, light_dir, save_video_path=None, fps=30)
        img = img_list[0]
        
        if save_img_path is not None:
            imageio.imwrite(save_img_path, img)
        
        return img
    
    @torch.no_grad()
    def render_to_video(self, vertices, faces, color, camera, light_dir, save_video_path=None, fps=30, batch_size=100):
        """
        渲染视频（批量处理多帧）
        
        Args:
            vertices: (T, N, 3) numpy array，T帧顶点数据
            faces: (F, 3) numpy array，面索引
            color: [R, G, B, A] 统一颜色
            camera: (4, 4) numpy array，相机变换矩阵
            light_dir: [x, y, z] 光照方向
            save_video_path: 可选，保存路径
            fps: 帧率
            batch_size: 每批处理的帧数，用于减少GPU内存使用
        
        Returns:
            img_list: 渲染的图像列表
        """
        assert vertices.ndim == 3 and vertices.shape[2] == 3
        assert faces.ndim == 2 and faces.shape[1] == 3
        
        T, N, _ = vertices.shape
        
        # 如果总帧数较大，分批处理以避免GPU内存溢出
        if T > batch_size:
            return self._render_to_video_batched(vertices, faces, color, camera, light_dir, save_video_path, fps, batch_size)
        
        # 如果帧数较少，使用原有逻辑一次性处理
        return self._render_batch(vertices, faces, color, camera, light_dir, save_video_path, fps)
    
    def _render_to_video_batched(self, vertices, faces, color, camera, light_dir, save_video_path, fps, batch_size):
        """
        分批渲染视频以避免GPU内存问题
        """
        T = vertices.shape[0]
        all_img_list = []
        
        # 分批处理
        num_batches = (T + batch_size - 1) // batch_size
        pbar = tqdm.tqdm(total=T, desc="Rendering[Nvdiffrast]")
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, T)
            
            # 提取当前批次的数据
            vertices_batch = vertices[start_idx:end_idx]
            
            # 渲染当前批次（不保存到文件）
            batch_img_list = self._render_batch(vertices_batch, faces, color, camera, light_dir, save_video_path=None, fps=fps)
            all_img_list.extend(batch_img_list)
            
            # 更新进度条
            pbar.update(end_idx - start_idx)
            
            # 清理GPU缓存
            torch.cuda.empty_cache()
        
        pbar.close()
        
        # 保存完整视频
        if save_video_path is not None:
            imageio.mimsave(save_video_path, all_img_list, fps=fps)
        
        return all_img_list
    
    def _render_batch(self, vertices, faces, color, camera, light_dir, save_video_path, fps):
        """
        渲染单个批次的帧
        """
        T, N, _ = vertices.shape
        
        # 转换到 torch 并移到 GPU
        vertices_torch = torch.from_numpy(vertices).float().cuda()  # (T, N, 3)
        faces_torch = torch.from_numpy(faces.astype(np.int32)).int().cuda()  # (F, 3)
        camera_torch = torch.from_numpy(camera.astype(np.float32)).float().cuda()  # (4, 4)
        
        # 归一化光照方向
        light_dir = np.asarray(light_dir, dtype=np.float32)
        light_dir = light_dir / (np.linalg.norm(light_dir) + 1e-8)
        light_dir_torch = torch.from_numpy(light_dir).float().cuda()  # (3,)
        
        # 基础颜色（归一化到 [0, 1]）并转换到线性空间
        # Pyrender 会在着色器中进行 sRGB 到线性空间的转换
        base_color = np.array(color[:3], dtype=np.float32) / 255.0
        # 转换到线性空间（gamma = 2.2）
        base_color = np.power(base_color, 2.2)
        base_color_torch = torch.from_numpy(base_color).float().cuda()  # (3,)
        
        # 将顶点转换到相机空间（批量处理）
        # 注意：camera 参数是相机的 pose（在世界坐标系中的位姿）
        # 需要取逆得到视图矩阵 V（从世界坐标到相机坐标的变换）
        V = torch.inverse(camera_torch)  # 视图矩阵
        
        vertices_homo = torch.cat([
            vertices_torch,
            torch.ones(T, N, 1, device='cuda')
        ], dim=2)  # (T, N, 4)
        
        # 批量矩阵乘法：(T, N, 4) @ V^T
        vertices_cam = torch.matmul(vertices_homo, V.T)  # (T, N, 4)
        vertices_cam_xyz = vertices_cam[..., :3]  # (T, N, 3)
        
        # ===== 构建正交投影矩阵（匹配 pyrender 的 OrthographicCamera）=====
        # pyrender 使用 OrthographicCamera(xmag=1.0, ymag=1.0)
        # 投影矩阵定义见 pyrender/camera.py 第 283-310 行
        height, width = self.resolution
        xmag = 1.0
        ymag = 1.0
        
        # 根据分辨率调整 xmag（匹配 pyrender 的逻辑）
        xmag = width / height * ymag
        
        # pyrender 的默认 znear 和 zfar
        znear = 0.05
        zfar = 100.0
        
        # 构建正交投影矩阵（OpenGL 标准）
        P = torch.zeros((4, 4), device='cuda', dtype=torch.float32)
        P[0, 0] = 1.0 / xmag
        P[1, 1] = 1.0 / ymag
        P[2, 2] = 2.0 / (znear - zfar)
        P[2, 3] = (zfar + znear) / (znear - zfar)
        P[3, 3] = 1.0
        
        # 计算每个面的法向量（在相机空间中，用于光照计算）
        v0 = vertices_cam_xyz[:, faces_torch[:, 0], :]  # (T, F, 3)
        v1 = vertices_cam_xyz[:, faces_torch[:, 1], :]  # (T, F, 3)
        v2 = vertices_cam_xyz[:, faces_torch[:, 2], :]  # (T, F, 3)
        
        edge1 = v1 - v0  # (T, F, 3)
        edge2 = v2 - v0  # (T, F, 3)
        
        # 叉积计算法向量
        normals = torch.cross(edge1, edge2, dim=2)  # (T, F, 3)
        
        # 归一化法向量
        normals = torch.nn.functional.normalize(normals, dim=2, eps=1e-8)  # (T, F, 3)
        
        # 计算光照强度（Lambert 模型）
        # 注意：m_mesh_renderer_pyrender.py 第34行使用 DirectionalLight(intensity=4.0)
        # 但 pyrender 使用 PBR 材质和 gamma correction (pow(color, 1/2.2))
        # 经过实验和分析，使用 intensity=1.15 能最佳匹配 pyrender 的视觉效果
        light_intensities = torch.clamp(
            -torch.sum(normals * light_dir_torch[None, None, :], dim=2) * 1.15,  # (T, F)
            0, 1
        )
        
        # 将面光照强度分配到顶点（使用 scatter_add 优化）
        vertex_intensities = torch.zeros(T, N, device='cuda')  # (T, N)
        vertex_counts = torch.zeros(T, N, device='cuda')  # (T, N)
        
        # 将面索引展开为所有顶点索引
        face_indices_flat = faces_torch.flatten().long()  # (F*3,) 转换为 int64
        
        # 将光照强度重复3次（每个面的3个顶点）
        light_intensities_repeated = light_intensities.repeat_interleave(3, dim=1)  # (T, F*3)
        
        # 使用 scatter_add 进行批量累加
        vertex_intensities.scatter_add_(1, face_indices_flat[None, :].expand(T, -1), light_intensities_repeated)
        vertex_counts.scatter_add_(1, face_indices_flat[None, :].expand(T, -1), torch.ones_like(light_intensities_repeated))
        
        vertex_intensities = vertex_intensities / (vertex_counts + 1e-8)  # (T, N)
        
        # 计算顶点颜色（应用光照）
        vertex_colors = vertex_intensities[..., None] * base_color_torch[None, None, :]  # (T, N, 3)
        
        # 应用 Gamma correction (匹配 pyrender 的 pow(color, 1.0/2.2))
        # Pyrender 在着色器中应用 gamma correction，见 mesh.frag 第 446 行
        vertex_colors = torch.pow(torch.clamp(vertex_colors, 0.0, 1.0), 1.0 / 2.2)
        
        # ===== 应用投影矩阵变换到裁剪空间 =====
        # 将相机空间坐标通过投影矩阵变换到裁剪空间
        vertices_cam_homo = torch.cat([
            vertices_cam_xyz,
            torch.ones(T, N, 1, device='cuda')
        ], dim=2)  # (T, N, 4)
        
        # 应用投影矩阵：(T, N, 4) @ P^T
        pos_clip_homo = torch.matmul(vertices_cam_homo, P.T)  # (T, N, 4)
        
        # 批量光栅化
        rast_out, _ = dr.rasterize(
            self.glctx,
            pos_clip_homo,
            faces_torch,
            resolution=[self.resolution[0], self.resolution[1]]
        )  # (T, H, W, 4)
        
        # 批量插值颜色
        color_out, _ = dr.interpolate(vertex_colors, rast_out, faces_torch)  # (T, H, W, 3)
        
        # 应用 alpha mask（背景为黑色）
        # 应用 alpha mask 并设置自定义背景颜色
        alpha_mask = (rast_out[..., 3:] > 0).float()  # (T, H, W, 1)
        
        color_out = color_out * alpha_mask + self.background_color[None, None, None, :] * (1 - alpha_mask)
        
        # alpha_mask = (rast_out[..., 3:] > 0).float()  # (T, H, W, 1)
        # color_out = color_out * alpha_mask  # (T, H, W, 3)
        
        # 转换为 numpy（注意：nvdiffrast 的输出是 bottom-up，需要垂直翻转）
        color_out_np = color_out.detach().cpu().numpy()  # (T, H, W, 3)
        
        # 生成图像列表
        img_list = []
        for i in range(T):
            img = color_out_np[i, ::-1, :, :]  # 垂直翻转
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
            img_list.append(img)
        
        if save_video_path is not None:
            imageio.mimsave(save_video_path, img_list, fps=fps)
        
        return img_list
