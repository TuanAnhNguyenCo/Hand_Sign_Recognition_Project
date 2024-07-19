<h2>Introduction</h2>
<p>- In this project, I have built a model to convert Vietnamese hand signs into words to help people who are deaf communicate with those who don't understand sign language</p>
<p>- I assume that words in a long video will be separated from each other by the action of keeping the arms straight down. </p>
<p>- Pipeline: </p>
<ul>
  <li> Firstly, a video that includes many signs will be separated into individual videos, each corresponding to a single sign, by my algorithm.</li>
  <li> After that, those videos will be passed into the model to predict the corresponding words.</li>
</ul>
<h2>Installation</h2>
<ul>
  <li> Download mmdeploy git:  `git clone https://github.com/open-mmlab/mmdeploy.git`</li>
  <li> Download pre-trained weights: 
    <ul> 
       <li> cd inference_utils</li>
       <li>Download weights: `gdown 1S9OOxt39vxk_ncZg9GAhUrx8FwFALcmm`</li>
    </ul>
  </li>
 </ul>
<h2>Running Inference API</h2>
 <li> `gunicorn fastAPI:app --bind 0.0.0.0:9091 --worker-class uvicorn.workers.UvicornWorker --timeout 300`</li>
<h2>Vietnamese Sign Language dataset</h2>
 <li> Comming Soon ... </li>
<h2>Training</h2>
  <li> `python3 main.py --config config/vtn_hc.yaml`</li>

