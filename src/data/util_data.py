import torch
import torch.nn.functional as F


def get_dataloader(dataset, split, batch_size=32, num_workers=4):
    if split == 'train':
        shuffle = True
        drop_last = True
    else:
        shuffle = False
        drop_last = False

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, drop_last=drop_last,
        pin_memory=True,
    )
    return loader


def split_data_to_chunks(data, chunk_size):
    chunks = []
    n = data.shape[0]

    for i in range(0, n, chunk_size):
        chunk = data[i:i+chunk_size]

        # 如果最后一个 chunk 长度不够，就在时间维度（dim=0）后面 padding 到 chunk_size
        if chunk.shape[0] < chunk_size:
            pad_frames = chunk_size - chunk.shape[0]

            # 对 [T, D] 或更高维数据，在第 0 维后面补 pad_frames 帧
            if data.ndim == 2:
                chunk = F.pad(chunk, (0, 0, 0, pad_frames),
                              mode='constant', value=0)
            elif data.ndim == 1:
                chunk = F.pad(chunk, (0, pad_frames), mode='constant', value=0)
            else:
                raise ValueError(f'invalid data.ndim: {data.ndim}')

        chunks.append(chunk)
    return chunks
