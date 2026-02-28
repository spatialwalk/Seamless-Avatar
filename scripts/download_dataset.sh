DATASET_ROOT=./datasets/seamless_smplx_dataset

./scripts/hfd.sh xwshi/Seamless-Avatar-Smplx-150h --dataset --exclude "flame_npz_annos.tar" --local-dir $DATASET_ROOT


# extract the dataset
for tar_file in $DATASET_ROOT/*.tar; do
    echo "Extracting $tar_file..."
    tar -xf "$tar_file" -C "$DATASET_ROOT" && echo "Extracted $tar_file successfully." && rm "$tar_file"
done