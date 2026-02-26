import shlex
import subprocess
import json
from pathlib import Path
import tempfile
import imageio
import numpy as np
import cv2
import os


def combine_two_audio(audio_a, audio_b, output):
    os.makedirs(os.path.dirname(output), exist_ok=True)
    if not os.path.exists(audio_a) or not os.path.exists(audio_b):
        raise FileNotFoundError(f"音频文件不存在: {audio_a} 或 {audio_b}")
    cmd = f"ffmpeg -i {audio_a} -i {audio_b} -filter_complex amix=inputs=2:duration=longest {output} -y -loglevel quiet"
   
    result = subprocess.run(shlex.split(cmd), capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg 命令执行失败，返回码: {result.returncode}")
   
    return output

def combine_video_and_audio(
    video_file, audio_file, output, quality=17, copy_audio=False
):
    os.makedirs(os.path.dirname(output), exist_ok=True)
    
    # 如果输入视频文件和输出文件路径相同，使用临时文件避免冲突
    if os.path.abspath(video_file) == os.path.abspath(output):
        # 创建临时文件用于中间处理
        temp_dir = os.path.dirname(output)
        temp_name = os.path.splitext(os.path.basename(output))[0] + "_temp" + os.path.splitext(output)[1]
        temp_output = os.path.join(temp_dir, temp_name)
    else:
        temp_output = output
    
    # audio_codec = "-c:a copy" if copy_audio else "-c:a aac"
    audio_codec = "-c:a copy" if copy_audio else "-c:a flac -strict -2"
    
    # 使用更安全的映射方式：
    # -map 0:v:0 明确选择第一个视频流
    # -map 1:a:0 明确选择音频文件的第一个音频流
    # 这样即使原视频没有音频流也不会出错
    cmd = (
        f"ffmpeg -i {video_file} -i {audio_file} -c:v libx264 -crf {quality} -pix_fmt yuv420p "
        f"-map 0:v:0 -map 1:a:0 {audio_codec} -shortest -y -hide_banner -loglevel error {temp_output}"
    )
    
    result = subprocess.run(shlex.split(cmd), capture_output=True, text=True)
    if result.returncode != 0:
        # 如果使用了临时文件且失败了，清理临时文件
        if temp_output != output and os.path.exists(temp_output):
            os.remove(temp_output)
        error_msg = f"ffmpeg 命令执行失败，返回码: {result.returncode}"
        if result.stderr:
            error_msg += f"\n错误信息: {result.stderr}"
        raise RuntimeError(error_msg)
    
    # 如果使用了临时文件，将其重命名为最终输出文件
    if temp_output != output:
        if os.path.exists(output):
            os.remove(output)
        os.rename(temp_output, output)

def combine_frames_and_audio(frame_files, audio_file, fps, output, quality=17):
    cmd = (
        f"ffmpeg -framerate {fps} -i {frame_files} -i {audio_file} -c:v libx264 -crf {quality} -pix_fmt yuv420p "
        f"-c:a copy -fflags +shortest -y -hide_banner -loglevel error {output}"
    )
    assert subprocess.run(shlex.split(cmd)).returncode == 0


def convert_video(video_file, output, quality=17):
    cmd = (
        f"ffmpeg -i {video_file} -c:v libx264 -crf {quality} -pix_fmt yuv420p "
        f"-fflags +shortest -y -hide_banner -loglevel error {output}"
    )
    assert subprocess.run(shlex.split(cmd)).returncode == 0


def resize_video(
    video_file, output, scale=None, width=None, height=None, quality=23, copy_audio=True
):
    """
    降低视频的分辨率以减小文件大小

    参数:
        video_file: 输入视频文件路径
        output: 输出视频文件路径
        scale: 缩放比例，例如0.5表示缩小到原来的一半
        width: 目标宽度，如果只指定宽度，高度会按比例缩放
        height: 目标高度，如果只指定高度，宽度会按比例缩放
        quality: CRF质量值 (范围0-51，值越高质量越低但文件越小，23是默认值)
        copy_audio: 是否直接复制音频而不重新编码

    注意: scale, width, height至少需要指定一个参数。优先级为：
    - 如果指定了scale，则忽略width和height
    - 如果同时指定了width和height，则精确缩放到该分辨率
    - 如果只指定了width或height，则保持宽高比进行缩放
    """
    # 构建缩放滤镜
    if scale is not None:
        filter_str = f"scale=iw*{scale}:ih*{scale}"
    elif width is not None and height is not None:
        filter_str = f"scale={width}:{height}"
    elif width is not None:
        filter_str = f"scale={width}:-1"  # 高度自动按比例缩放
    elif height is not None:
        filter_str = f"scale=-1:{height}"  # 宽度自动按比例缩放
    else:
        raise ValueError("必须指定scale、width或height参数中的至少一个")

    # 确定音频处理方式
    audio_codec = "-c:a copy" if copy_audio else "-c:a aac -b:a 128k"

    # 构建命令
    cmd = (
        f'ffmpeg -i {video_file} -vf "{filter_str}" -c:v libx264 -crf {quality} -preset medium '
        f"-pix_fmt yuv420p {audio_codec} -y -hide_banner -loglevel error {output}"
    )
    assert subprocess.run(shlex.split(cmd)).returncode == 0


def reencode_audio(audio_file, output):
    cmd = f"ffmpeg -i {audio_file} -y -hide_banner -loglevel error {output}"
    assert subprocess.run(shlex.split(cmd)).returncode == 0


def extract_frames(filename, output_dir, quality=1):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = (
        f"ffmpeg -i {filename} -qmin 1 -qscale:v {quality} -y -start_number 0 -hide_banner -loglevel error "
        f'{output_dir / "%06d.jpg"}'
    )
    assert subprocess.run(shlex.split(cmd)).returncode == 0


def concatenate_videos_horizontally(
    left_video, right_video, output, use_audio_from="left", quality=17, text_list=None, fontsize=36, target_height=None
):
    """
    将两个视频水平拼接在一起

    参数:
        left_video: 左侧视频文件路径
        right_video: 右侧视频文件路径
        output: 输出视频文件路径
        use_audio_from: 使用哪个视频的音频 ("left" 或 "right")
        quality: CRF质量值 (值越低质量越高，文件越大)
        text_list: 可选，包含两个字符串的列表 [左视频文本, 右视频文本]，用于标注视频
        fontsize: 标注文本的字体大小，默认为36
        target_height: 输出视频的目标高度，如果不指定，将使用第一个视频的高度
    """
    # 获取视频的高度信息
    def get_video_height(video_file):
        cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 "{video_file}"'
        result = subprocess.run(shlex.split(cmd), capture_output=True, text=True)
        if result.returncode == 0:
            return int(result.stdout.strip())
        return None
    
    left_height = get_video_height(left_video)
    right_height = get_video_height(right_video)
    
    # 如果未指定目标高度，使用左视频的高度
    if target_height is None and left_height is not None:
        target_height = left_height
    
    # 确保我们有一个有效的目标高度
    if target_height is None:
        raise ValueError("无法确定目标高度，请手动指定 target_height 参数")
    
    audio_map = "0:a" if use_audio_from == "left" else "1:a"
    
    # 构建滤镜命令
    if text_list and len(text_list) == 2:
        # 为每个视频添加文本标注及调整尺寸
        filter_complex = (
            f"[0:v]scale=-1:{target_height},drawtext=text='{text_list[0]}':fontcolor=white:fontsize={fontsize}:box=1:"
            f"boxcolor=black@0.5:boxborderw=5:x=10:y=10[left];"
            f"[1:v]scale=-1:{target_height},drawtext=text='{text_list[1]}':fontcolor=white:fontsize={fontsize}:box=1:"
            f"boxcolor=black@0.5:boxborderw=5:x=10:y=10[right];"
            f"[left][right]hstack=inputs=2[v]"
        )
    else:
        # 原始无文本版本，但仍然调整尺寸
        filter_complex = (
            f"[0:v]scale=-1:{target_height}[left];"
            f"[1:v]scale=-1:{target_height}[right];"
            f"[left][right]hstack=inputs=2[v]"
        )
    
    cmd = (
        f'ffmpeg -i {left_video} -i {right_video} -filter_complex "{filter_complex}" '
        f'-map "[v]" -map {audio_map} -c:v libx264 -crf {quality} -pix_fmt yuv420p '
        f"-c:a copy -y -hide_banner -loglevel error {output}"
    )
    assert subprocess.run(shlex.split(cmd)).returncode == 0


def extract_audio(video_file, output, audio_format=None):
    """
    从视频文件中提取音频

    参数:
        video_file: 视频文件路径
        output: 输出音频文件路径
        audio_format: 输出音频格式 (例如 "mp3", "aac", "wav")，如果为None则从输出文件扩展名推断
    """
    if audio_format is None:
        # 从输出文件名推断格式
        audio_format = Path(output).suffix.lstrip(".")

    # 根据格式选择适当的编码器
    if audio_format.lower() == "mp3":
        codec = "libmp3lame"
    elif audio_format.lower() == "aac":
        codec = "aac"
    elif audio_format.lower() == "wav":
        codec = "pcm_s16le"
    else:
        # 对于其他格式，尝试使用格式名作为编码器
        codec = audio_format

    # 不再使用copy模式，而是始终重新编码
    cmd = f"ffmpeg -i {video_file} -vn -acodec {codec} -y -hide_banner -loglevel error {output}"
    assert subprocess.run(shlex.split(cmd)).returncode == 0


def get_video_info(video_file):
    """
    获取视频文件的详细信息
    
    参数:
        video_file: 视频文件路径
        
    返回:
        包含视频信息的字典，包括:
        - 视频流信息 (分辨率、帧率、编码器、总帧数等)
        - 音频流信息 (采样率、声道数、编码器等) 
        - 总时长
        - 文件大小
        - 比特率
    """
    # 获取文件基本信息
    cmd = (
        f'ffprobe -v quiet -print_format json -show_format -show_streams "{video_file}"'
    )
    result = subprocess.run(shlex.split(cmd), capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"获取视频信息失败: {result.stderr}")
    
    info = json.loads(result.stdout)
    
    # 构造更友好的返回结果
    video_info = {
        "filename": Path(video_file).name,
        "filesize_mb": (
            round(float(info["format"]["size"]) / (1024 * 1024), 2)
            if "size" in info["format"]
            else None
        ),
        "duration": (
            float(info["format"]["duration"]) if "duration" in info["format"] else None
        ),
        "bitrate_kbps": (
            int(int(info["format"]["bit_rate"]) / 1000)
            if "bit_rate" in info["format"]
            else None
        ),
        "video": {},
        "audio": {},
    }
    
    # 查找视频和音频流
    for stream in info["streams"]:
        if stream["codec_type"] == "video":
            # 尝试直接获取总帧数
            nb_frames = stream.get("nb_frames")
            
            # 如果没有直接提供帧数信息，则计算估算值
            if nb_frames is None or nb_frames == "N/A":
                if "duration" in stream and "r_frame_rate" in stream:
                    duration = float(stream["duration"])
                    fps = eval(stream["r_frame_rate"])
                    nb_frames = int(duration * fps)
                elif "duration" in info["format"] and "r_frame_rate" in stream:
                    duration = float(info["format"]["duration"])
                    fps = eval(stream["r_frame_rate"])
                    nb_frames = int(duration * fps)
                else:
                    nb_frames = None
            else:
                nb_frames = int(nb_frames)
            
            video_info["video"] = {
                "resolution": f"{stream.get('width', 'N/A')}x{stream.get('height', 'N/A')}",
                "width": stream.get("width"),
                "height": stream.get("height"),
                "codec": stream.get("codec_name"),
                "pixel_format": stream.get("pix_fmt"),
                "fps": (
                    round(eval(stream["r_frame_rate"]), 2)
                    if "r_frame_rate" in stream
                    else None
                ),
                "bitrate_kbps": (
                    int(int(stream["bit_rate"]) / 1000)
                    if "bit_rate" in stream
                    else None
                ),
                "total_frames": nb_frames,
            }
        elif stream["codec_type"] == "audio":
            video_info["audio"] = {
                "codec": stream.get("codec_name"),
                "sample_rate": stream.get("sample_rate"),
                "channels": stream.get("channels"),
                "bitrate_kbps": (
                    int(int(stream["bit_rate"]) / 1000)
                    if "bit_rate" in stream
                    else None
                ),
            }
    
    return video_info


def print_video_info(video_file):
    """
    打印视频文件的关键信息
    
    参数:
        video_file: 视频文件路径
    """
    try:
        info = get_video_info(video_file)
        
        print(f"文件名: {info['filename']}")
        print(f"文件大小: {info['filesize_mb']} MB")
        print(f"时长: {info['duration']:.2f} 秒")
        print(f"总比特率: {info['bitrate_kbps']} kbps")
        
        if info["video"]:
            print("\n视频流:")
            print(f"  分辨率: {info['video']['resolution']}")
            print(f"  帧率: {info['video']['fps']} fps")
            if info['video']['total_frames'] is not None:
                print(f"  总帧数: {info['video']['total_frames']}")
            print(f"  编码: {info['video']['codec']}")
            print(f"  视频比特率: {info['video']['bitrate_kbps'] or 'N/A'} kbps")
        
        if info["audio"]:
            print("\n音频流:")
            print(f"  编码: {info['audio']['codec']}")
            print(f"  采样率: {info['audio']['sample_rate']} Hz")
            print(f"  声道数: {info['audio']['channels']}")
            print(f"  音频比特率: {info['audio']['bitrate_kbps'] or 'N/A'} kbps")
    
    except Exception as e:
        print(f"获取视频信息失败: {e}")



def stitch_videos(video_paths, output_path, video_per_row=None, target_size=None, resize_method='fit', audio_source_idx=None, text_list=None):
    '''
    将多个视频拼接成网格布局的单个视频，支持不同分辨率视频

    Args:
        video_paths: list of video paths - 输入视频路径列表
        output_path: output video path - 输出视频路径
        video_per_row: int - 每行视频的数量，默认为视频总数
        target_size: tuple - 每个视频的目标大小(宽,高)，默认为None(使用最大尺寸)
        resize_method: str - 调整尺寸的方法，'fit'(保持比例缩放) 或 'stretch'(拉伸)
        audio_source_idx: int - 指定使用哪个视频的音频，默认为None(不添加音频)
        text_list: list of str - 每个视频的文字描述，默认为None(不添加文字)
    '''
    if video_per_row is None:
        video_per_row = len(video_paths)

    # 读取所有视频
    readers = [imageio.get_reader(path) for path in video_paths]

    # 获取第一个视频的帧率作为参考
    fps = readers[0].get_meta_data()['fps']
    
    # 获取所有视频的分辨率
    video_sizes = []
    for reader in readers:
        first_frame = reader.get_data(0)
        video_sizes.append(first_frame.shape[:2])  # (height, width)
    
    # 如果没有指定目标尺寸，使用所有视频中的最大宽度和最大高度
    if target_size is None:
        max_height = max(size[0] for size in video_sizes)
        max_width = max(size[1] for size in video_sizes)
        target_height, target_width = max_height, max_width
    else:
        target_width, target_height = target_size  # 注意参数顺序是(宽,高)
    
    # 计算网格布局
    n_videos = len(video_paths)
    n_rows = (n_videos + video_per_row - 1) // video_per_row
    n_cols = min(n_videos, video_per_row)
    
    # 创建临时无音频输出文件路径
    temp_output_path = output_path
    if audio_source_idx is not None:
        base_dir = os.path.dirname(output_path)
        base_name = os.path.splitext(os.path.basename(output_path))[0]
        ext = os.path.splitext(output_path)[1]
        temp_output_path = os.path.join(base_dir, f"{base_name}_temp{ext}")
    
    # 创建输出视频写入器
    writer = imageio.get_writer(temp_output_path, fps=fps)
    
    try:
        # 初始化帧计数器
        frame_idx = 0
        while True:
            # 创建组合帧
            combined_frame = np.zeros(
                (target_height * n_rows, target_width * n_cols, 3), dtype=np.uint8)
            
            # 标记是否所有视频都已结束
            all_videos_ended = True
            
            # 读取每个视频的当前帧
            for idx, reader in enumerate(readers):
                try:
                    frame = reader.get_data(frame_idx)
                    h, w = frame.shape[:2]
                    
                    # 根据resize_method调整帧的大小
                    if resize_method == 'fit':
                        # 保持宽高比缩放
                        scale = min(target_width / w, target_height / h)
                        new_w, new_h = int(w * scale), int(h * scale)
                        resized_frame = cv2.resize(frame, (new_w, new_h))
                        
                        # 创建黑色背景
                        fitted_frame = np.zeros((target_height, target_width, 3), dtype=np.uint8)
                        # 将调整大小后的帧居中放置
                        y_offset = (target_height - new_h) // 2
                        x_offset = (target_width - new_w) // 2
                        fitted_frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_frame
                        processed_frame = fitted_frame
                    else:  # 'stretch'
                        # 直接拉伸到目标大小
                        processed_frame = cv2.resize(frame, (target_width, target_height))
                    
                    # 如果提供了文本列表，添加文本到视频帧
                    if text_list is not None and idx < len(text_list) and text_list[idx]:
                        # 添加文本到中间顶部
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.8
                        font_thickness = 2
                        text_color = (255, 255, 255)  # 白色文字
                        bg_color = (0, 0, 0)  # 黑色背景
                        
                        # 获取文本框的大小
                        text = text_list[idx]
                        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
                        
                        # 计算文本居中的x坐标
                        x_position = (target_width - text_width) // 2
                        
                        # 添加黑色背景
                        cv2.rectangle(processed_frame, (x_position - 5, 10), (x_position + text_width + 5, 10 + text_height + 10), bg_color, -1)
                        
                        # 添加白色文字
                        cv2.putText(processed_frame, text, (x_position, 10 + text_height), font, font_scale, text_color, font_thickness)
                    
                    # 将处理后的帧放入对应位置
                    row = idx // video_per_row
                    col = idx % video_per_row
                    combined_frame[row*target_height:(row+1)*target_height,
                                  col*target_width:(col+1)*target_width] = processed_frame
                    all_videos_ended = False
                except (IndexError, RuntimeError):
                    # 如果某个视频已经结束，保持黑色帧填充
                    pass
            
            # 如果所有视频都结束了，退出循环
            if all_videos_ended:
                break
            
            # 写入合并后的帧
            writer.append_data(combined_frame)
            frame_idx += 1
    
    except (IndexError, RuntimeError):
        # 当所有视频都处理完毕时退出
        pass
    
    finally:
        # 清理资源
        writer.close()
        for reader in readers:
            reader.close()
    
    # 如果指定了音频源，添加音频到拼接后的视频
    if audio_source_idx is not None and 0 <= audio_source_idx < len(video_paths):
        import subprocess
        import tempfile

        final_output_path = output_path # Final path should be the original output path

        # 检查临时文件是否与最终输出文件路径相同，如果是，则需要一个新的临时音频文件路径
        # 并将最终视频先输出到另一个临时路径，最后再重命名
        temp_audio_path = None
        final_temp_video_path = temp_output_path # Path of the video generated by imageio

        if temp_output_path == output_path:
            # Avoid overwriting the source video if it's also the output
            base_dir = os.path.dirname(output_path)
            base_name = os.path.splitext(os.path.basename(output_path))[0]
            ext = os.path.splitext(output_path)[1]
            # Use a different temporary path for the final merged video before renaming
            final_temp_video_path = os.path.join(base_dir, f"{base_name}_final_temp{ext}")


        with tempfile.NamedTemporaryFile(suffix='.aac', delete=False) as tmp_audio_file:
            temp_audio_path = tmp_audio_file.name

        try:
            # 计算视频时长
            duration = frame_idx / fps
            audio_source_video = video_paths[audio_source_idx]

            # 1. 使用 ffmpeg 提取精确时长的音频
            #    -ss 0: 从头开始提取
            #    -t duration: 提取指定时长
            #    -i audio_source_video: 输入源视频
            #    -vn: 不包含视频
            #    -acodec copy: 直接复制音频流，避免重编码损失 (如果源格式兼容)
            #    或者 -acodec aac -q:a 2: 使用aac编码 (更通用)
            #    -y: 覆盖输出文件
            extract_cmd = [
                'ffmpeg', '-y',
                '-i', audio_source_video,
                '-ss', '0',
                '-t', str(duration),
                '-vn',       # No video output
                '-acodec', 'aac', # Encode to aac for wider compatibility
                '-strict', '-2',  # Necessary for some AAC experimental encoders
                temp_audio_path
            ]
            print(f"Running ffmpeg to extract audio: {' '.join(extract_cmd)}")
            subprocess.run(extract_cmd, check=True, capture_output=True)

            # 检查提取的音频文件是否存在且非空
            if not os.path.exists(temp_audio_path) or os.path.getsize(temp_audio_path) == 0:
                raise RuntimeError(f"Failed to extract audio from {audio_source_video}")


            # 2. 使用 ffmpeg 合并视频和提取的音频
            #    -i final_temp_video_path: 输入视频 (imageio生成的)
            #    -i temp_audio_path: 输入音频 (ffmpeg提取的)
            #    -c:v copy: 直接复制视频流
            #    -c:a copy: 直接复制音频流 (这里是aac)
            #    -shortest: 以最短的输入流结束输出 (关键参数)
            #    -y: 覆盖输出文件
            #    final_output_path: 输出文件路径 (可能是最终路径或临时路径)
            merge_target = final_temp_video_path if temp_output_path == output_path else output_path
            merge_cmd = [
                'ffmpeg', '-y',
                '-i', temp_output_path, # Video generated by imageio
                '-i', temp_audio_path,  # Audio extracted by ffmpeg
                '-c:v', 'copy',         # Copy video stream
                '-c:a', 'aac',          # Re-encode audio stream if needed, or copy if already AAC
                '-strict', '-2',
                '-shortest',            # Finish encoding when the shortest input stream ends
                merge_target
            ]
            print(f"Running ffmpeg to merge video and audio: {' '.join(merge_cmd)}")
            subprocess.run(merge_cmd, check=True, capture_output=True)

            # 如果之前使用了临时的最终视频路径，现在重命名
            if temp_output_path == output_path:
                 if os.path.exists(output_path):
                    os.remove(output_path) # Remove original if it exists
                 os.rename(final_temp_video_path, output_path)


            print(f"Successfully merged video and audio to: {output_path}")
            return output_path

        except subprocess.CalledProcessError as e:
            print(f"ffmpeg command failed: {e}")
            print(f"Stderr: {e.stderr.decode()}")
            print(f"Stdout: {e.stdout.decode()}")
            # 如果合并失败，但无音频视频已生成，保留无音频视频
            if os.path.exists(temp_output_path) and temp_output_path != output_path:
                 # Ensure the original output path doesn't contain a failed merge attempt
                if os.path.exists(output_path):
                    os.remove(output_path)
                os.rename(temp_output_path, output_path)
                print(f"Audio merge failed, saved video without audio at: {output_path}")
                return output_path
            elif os.path.exists(temp_output_path) and temp_output_path == output_path:
                 print(f"Audio merge failed. Original file {output_path} (no audio) might be kept or overwritten by failed attempt.")
                 return output_path # Return the path, even if potentially broken
            else:
                 print(f"Audio merge failed and intermediate video file {temp_output_path} not found.")
                 return None # Indicate failure

        except Exception as e:
            print(f"An unexpected error occurred during audio processing: {e}")
             # Fallback similar to CalledProcessError
            if os.path.exists(temp_output_path) and temp_output_path != output_path:
                if os.path.exists(output_path):
                    os.remove(output_path)
                os.rename(temp_output_path, output_path)
                return output_path
            elif os.path.exists(temp_output_path) and temp_output_path == output_path:
                return output_path
            else:
                return None

        finally:
            # 清理临时文件
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            # If the original temp_output_path was different from output_path and still exists (merge failed early/or merge target was different), remove it.
            if temp_output_path != output_path and os.path.exists(temp_output_path):
                 # Double check it wasn't the final renamed output in case of failure
                 if not os.path.exists(output_path) or os.path.abspath(temp_output_path) != os.path.abspath(output_path):
                      os.remove(temp_output_path)


    # 如果没有指定音频源，或者处理音频失败但保留了无音频文件
    elif os.path.exists(temp_output_path):
         # If temp path is different from final path, rename it
         if temp_output_path != output_path:
              if os.path.exists(output_path):
                    os.remove(output_path)
              os.rename(temp_output_path, output_path)
         return output_path
    else:
        print("Stitching finished, but no output file was generated.")
        return None # Indicate failure if no file exists

if __name__ == "__main__":
    # video_path = "/root/autodl-fs/VHAP_track/VHAP_track_monocular/mono_jp/20250418_large_head_emo-split/20250418_large_head_emo-split.mp4"
    # extract_audio(
    #    video_path,
    #     "test.mp3",
    # )
    # concatenate_videos_horizontally(
    #     '../result/jp_raw_720p.mp4',
    #     '../result/jp.mp4',
    #     '../result/jp_compare.mp4',
    #     use_audio_from='left',
    #     text_list=['raw','spectre flame driven'],
    #     fontsize=72,
    # )
    # combine_video_and_audio(
    #     '../result/jp.mp4',
    #     '../result/jp.mp3',
    #     '../result/jp_with_audio.mp4',
    #     quality=17,
    #     copy_audio=True
    # )
    
    print_video_info('../result/jp_compare.mp4')