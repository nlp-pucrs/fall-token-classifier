# fall-token-classifier
Fall Detection in Clinical Notes using Language Models and Token Classifier

**Author:** Joaquim Santos, Henrique D. P. dos Santos and Renata Vieira

**Abstract:** Electronic  health  records  (EHR)  are  a  key  sourceof information to identify adverse events in patients. The largestcategory  of  adverse  events  in  hospitals  is  fall  incidents.  Theidentification  of  such  incidents  guide  to  a  better  comprehensionof  the  event  and  enhance  the  quality  of  patient  health  care.In  this  initial  work,  we  compare  the  performance  of  Sentence-Classifier (StC) against the Token-Classifier (TkC) with state-of-the-art recurrent neural networks (RNN) to detect fall incidentsin  progress  notes.  Our  experiments  show  that  the  use  of  deep-learning algorithms as token-classifier outperforms text-classifier.It improves fall identification using StC from 65% to 92% with TkC  (F-Measure).  Additionally,  the  token  classifier  is  able  toexplain  which  words  are  most  important  in  positive  detection.

**Keywords:** Fall Detection, Clinical Notes, Language Model, Token  Classifier

Full Text, BibText

### Online Experiments
##### Run our experiments online with Binder
You need install Flair 0.4.3 ```pip install flair==0.4.3```
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/nlp-pucrs/fall-token-classifier/master)

### PUCRS A.I. in HealthCare
This project belongs to [GIAS at PUCRS, Brazil](http://www.inf.pucrs.br/ia-saude/)
