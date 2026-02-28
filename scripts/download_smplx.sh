wget https://huggingface.co/lilpotat/pytorch3d/resolve/main/SMPLX_NEUTRAL_2020.npz -O models/smplx/SMPLX_NEUTRAL_2020.npz


# urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }
# COLOR='\033[0;32m'

# # username and password input

# echo -e "\nIf you do not have an account you can register at https://smpl-x.is.tue.mpg.de/ following the installation instruction."
# read -p "Username (smpl-x):" username
# read -p "Password (smpl-x):" password
# username=$(urle $username)
# password=$(urle $password)

# mkdir -p models/smplx

# wget --post-data "username=$username&password=$password" \
# 'https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=SMPLX_NEUTRAL_2020.npz&resume=1' \
# -O models/smplx/SMPLX_NEUTRAL_2020.npz \
# --no-check-certificate --continue