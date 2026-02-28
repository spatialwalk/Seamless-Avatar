echo -e "\nIf you do not have an account you can register at https://smpl-x.is.tue.mpg.de/following the installation instruction."
read -p "Username (smpl-x):" username
read -p "Password (smpl-x):" password
username=$(urle $username)
password=$(urle $password)

mkdir -p model/smplx

wget --post-data "username=$username&password=$password" \
'https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=SMPLX_NEUTRAL_2020.npz&resume=1' \
-O model/smplx/SMPLX_NEUTRAL_2020.npz \
--no-check-certificate --continue