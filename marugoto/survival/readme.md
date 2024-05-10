# Marugoto Survival Prediction Branch

Welcome to the survival prediction branch of Marugoto! This branch introduces some differences compared to the original version. Please follow the instructions below to run the code. Compatibility updates will be provided shortly.

## Example Commands
### Train
```
python train.py \
-ct /path/to/clinical_table.csv \
-st /path/to/slide_table.csv \
-o /path/to/output_location \
-f /path/to/feature_directory \
-t OS OS_E DFS DFS_E
```
### Deploy
```
python eval.py \
-ct /path/to/clinical_table.csv \
-st /path/to/clinical_slide_table.xlsx \
-o /path/to/eval_results \
-f /path/to/feature_directory \
-m /path/to/model_output \
-c cohort_name \
-t OS OS_E DFS DFS_E
```
### Visualization 
```
python survival_visualizations.py \
-f /path/to/feature_directory \
-sd /path/to/slide_directory \
-m /path/to/model_output \
-sc /path/to/deploy_output \
-o /path/to/output_results  \
-nt 6 \
-np 10 \
-tl True \
-nf 1024 \
-g True
```


### Additional Information
```
ct = clini table, using format:|PATIENT|FILENAME|OS|OS_E|DFS|DFS_E|
st = slide table, using format:|PATIENT| (required but redundant as slide info read from ct)
o = output location
f = feature directory
t = stats: OS overall survival, OS_E os event (i.e. dead/alive), DFS disease free status, DFS_E DFS event
m = model path (location of .pth output from train.py script)
c = cohort (additional name for output of eval.py)
sd = slides directory (svs files)
sc = the Marugoto Survival Deploy output file (cohort_score.csv)
nt = number of tiles per patient to store
np = number of patient to plot
tl = a boolean representing whether top patients must be plotted, otherwise bottom patients
nf = number of features extracted (e.g 1024 UNI, 768 CTransPath)
g = boolean representing whether the geojson of the top tiles must be saved (geojson could be imported into QuPath to observe the context of the top tiles)
```
