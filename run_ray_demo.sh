##############################################################################################

DVCPROGRAM=./mbsoptflow
ANALYSISPROGRAM=./voxel2mesh

##############################################################################################
GPU=0

ALPHA=0.1
NORMALIZATION="histogram"
LEVEL=10
SCALE=0.9
PREFILTER=0.5

LOCALGLOBAL=3

ADDITIONAL_ARGS="--median --export_warp"

ROOTPATH="/Demos/RayDemo/"
MASK="/home/brunsste/Documents/MBS-3D-OptFlow/Demos/RayDemo/Mask01/"
OUTPATH="/optflow/"

##############################################################################################

EXPERIMENT_LIST="
Frame02
"

############################# Execute #############################
ARGUMENTS="-alpha "$ALPHA" -norm "$NORMALIZATION" -level "$LEVEL" -scale "$SCALE" -prefilter gaussian "$PREFILTER" -gpu0 "$GPU" "$ADDITIONAL_ARGS

for i in $EXPERIMENT_LIST; do
    INPATH=$i
    $DVCPROGRAM -i0 $ROOTPATH"Frame01/" -i1 $ROOTPATH$INPATH -m $MASK $ARGUMENTS
done
###################################################################
