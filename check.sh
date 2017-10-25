D='test'
[[ $# -eq 1 ]] && D=$1
./build/bin/dso_nogui  files=$D/images.zip calib=$D/camera.txt gamma=$D/pcalib.txt vignette=$D/vignette.png preset=0 mode=0 nogui=1 quiet=0
