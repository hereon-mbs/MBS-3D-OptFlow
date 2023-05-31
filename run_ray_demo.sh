##############################################################################################

DVCPROGRAM=./mbsoptflow
ANALYSISPROGRAM=./voxel2mesh

##############################################################################################
GPU=0

ALPHA=0.1
NORM="equalized"
LEVEL=10
SCALE=0.9
PREFILTER=0.5

LOCALGLOBAL=3

ADDITIONAL_ARGS="--median --export_warp"

FRAME0="/Demos/RayDemo/Frame01/"
FRAME1="/Demos/RayDemo/Frame02/"
MASK="none"
OUTPATH="/optflow/"

##############################################################################################

############################# Execute #############################
ARGUMENTS="-alpha "$ALPHA" -norm "$NORM" -level "$LEVEL" -scale "$SCALE" -prefilter gaussian "$PREFILTER" -gpu0 "$GPU" "$ADDITIONAL_ARGS
$DVCPROGRAM -i0 $FRAME0 -i1 $FRAME1 -m $MASK $ARGUMENTS
###################################################################



