import os
import numpy as np
import pandas as pd
from flair.data import Sentence
from flair.models import TextClassifier
from flair.models import SequenceTagger
from sklearn.metrics import classification_report, f1_score

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

class EvaluatesTokenClassification:

	def __init__(self, path_model_token, path_valid_data, output_file_name):
		self.path_model_token : str = path_model_token
		self.path_valid_data : str = path_valid_data
		self.output_file_name : str = output_file_name
		self.tagger = SequenceTagger.load(self.path_model_token)
	
	def load_validation_data(self):
		return pd.read_csv(self.path_valid_data, delimiter='\t')

	def predict_sentence(self, sentencesToPredict):
		pred = []
		y_pred, y_true = [], []
		sentencesToPredict = sentencesToPredict.to_numpy()
		new_file = open(self.output_file_name, "+w", encoding='utf8')
		new_file.write('sentence\ty_pred\ty_true\n')

		for sentence in sentencesToPredict:
			#if not np.isnan(sentence[1]): #Algumas evoluções estão sem classificação
			sentence_to_predict = Sentence(sentence[0])
			self.tagger.predict(sentence_to_predict)
			if sentence_to_predict.to_tagged_string().find('<B-QUEDA>') != -1:
				label = 1
			else:
				label = 0
			y_pred.append(label)
			y_true.append(int(sentence[-1]))
			new_file.write(str(sentence[0])+"\t"+str(label)+"\t"+str(sentence[-1])+'\n')
		new_file.close()

		target_names = ['class 0', 'class 1']

		print(classification_report(y_true, y_pred, target_names=target_names))
		print('\tF1 Score\n')
		print('Class 0\t', f1_score(y_true, y_pred, pos_label=0, average='binary'))
		print('Class 1\t', f1_score(y_true, y_pred, pos_label=1, average='binary'))

	def make_validation_token_classification(self):
		print(' ')
		print('--------------------------START VALIDATION TOKEN------------------------------')
		print(' ')

		data_valid = self.load_validation_data()

		print(' ')
		print(' --- VALID --- ')
		self.predict_sentence(data_valid)

class EvaluatesTextClassification:

	def __init__(self, path_model_text, path_valid_data, output_file_name):
		self.path_model_text : str = path_model_text
		self.path_valid_data : str = path_valid_data
		self.output_file_name : str = output_file_name
		self.classifier = TextClassifier.load(self.path_model_text)

	def predict_sentence(self, sentenceToPredict):
		sentence = Sentence(sentenceToPredict)
		self.classifier.predict(sentence)
		return sentence.labels

	def load_validation_data(self):
		return pd.read_csv(self.path_valid_data, delimiter='\t')

	def labeling_valid_data(self, dataValid):
		y_true, y_pred = [], []
		dataValid = dataValid.to_numpy()
		new_file = open(self.output_file_name, "+w", encoding='utf8')
		new_file.write('sentence\ty_pred\ty_true\n')

		for row in dataValid:
			#if not np.isnan(row[1]): #Algumas evoluções estão sem classificação
			predicted_label = int(str(self.predict_sentence(row[0])[0]).split(' ')[0])
			y_pred.append(predicted_label)
			y_true.append(int(row[1]))
			new_file.write(str(row[0])+"\t"+str(predicted_label)+"\t"+str(row[1])+'\n')
		new_file.close()

		target_names = ['class 0', 'class 1']
		print(classification_report(y_true, y_pred, target_names=target_names))
		print('\tF1 Score\n')
		print('Class 0\t', f1_score(y_true, y_pred, pos_label=0, average='binary'))
		print('Class 1\t', f1_score(y_true, y_pred, pos_label=1, average='binary'))

	def make_validation_text_classification(self):
		
		print(' ')
		print('--------------------------START VALIDATION TEXT------------------------------')
		print(' ')
		
		data_valid = self.load_validation_data()

		print(' ')
		print(' --- VALID --- ')
		self.labeling_valid_data(data_valid)


if __name__ == '__main__':

	'''
	INNER-DATASET
	
	
	print(color.BLUE + '--------------- (NER) EVAL INNER-DATASET ---------------' + color.END)
	
	tkc_inner = EvaluatesTokenClassification(
		path_model_token='/content/drive/My Drive/Colab Notebooks/Fall-Recognition/Experiments/NER-Fall/resources/taggers/example-ner/best-model.pt',
		path_valid_data='/content/drive/My Drive/Colab Notebooks/Fall-Recognition/Data/Text-Test.csv'
		)
	tkc_inner.make_validation_token_classification()

	print(' ')

	print(color.BLUE + '--------------- (TXC) EVAL INNER-DATASET ---------------' + color.END)
	
	txc_inner = EvaluatesTextClassification(
		path_model_text='/content/drive/My Drive/Colab Notebooks/Fall-Recognition/Experiments/TxC-Fall/resources/taggers/example-txc/best-model.pt',
		path_valid_data='/content/drive/My Drive/Colab Notebooks/Fall-Recognition/Data/Text-Test.csv'
		)
	txc_inner.make_validation_text_classification()

	'''
	
	'''
	print(color.BLUE + '--------------- (NER) EVAL OUTER-DATASET ---------------' + color.END)
	
	tkc_outer = EvaluatesTokenClassification(
		path_model_token='/content/drive/My Drive/Colab Notebooks/Fall-Recognition/Experiments/NER-Fall/resources/taggers/example-ner/best-model.pt',
		path_valid_data='/content/drive/My Drive/Colab Notebooks/Fall-Recognition/Data/Fall_Extra_Validation.csv',
		output_file_name='tkc_data_valid.csv'
		)
	tkc_outer.make_validation_token_classification()

	'''

	'''
	print(color.BLUE + '--------------- (TXC) EVAL OUTER-DATASET ---------------' + color.END)
	
	txc_outer = EvaluatesTextClassification(
		path_model_text='/content/drive/My Drive/Colab Notebooks/Fall-Recognition/Experiments/TxC-Fall/resources/taggers/example-txc/best-model.pt',
		path_valid_data='/content/drive/My Drive/Colab Notebooks/Fall-Recognition/Data/Fall_Extra_Validation.csv',
		output_file_name='txc_data_valid.csv'
		)
	txc_outer.make_validation_text_classification()
	'''	

	'''
	print(color.BLUE + '--------------- (NER) EVAL CROSS-VALIDATION FOLDER: 1 ---------------' + color.END)
	
	tkc_1 = EvaluatesTokenClassification(
		path_model_token='/content/drive/My Drive/Colab Notebooks/Fall-Recognition/Experiments/CROSS-NER/FOLDER_1/TRN-FD1/best-model.pt',
		path_valid_data='/content/drive/My Drive/Colab Notebooks/Fall-Recognition/Data/Fall_Extra_Validation.csv',
		output_file_name='tkc_data_valid_fold_1.csv'
		)
	tkc_1.make_validation_token_classification()

	print(' ')
	'''
	
	'''
	print(color.BLUE + '--------------- (NER) EVAL CROSS-VALIDATION FOLDER: 2 ---------------' + color.END)

	tkc_2 = EvaluatesTokenClassification(
		path_model_token='/content/drive/My Drive/Colab Notebooks/Fall-Recognition/Experiments/CROSS-NER/FOLDER_2/TRN-FD2/best-model.pt',
		path_valid_data='/content/drive/My Drive/Colab Notebooks/Fall-Recognition/Data/Fall_Extra_Validation.csv',
		output_file_name='tkc_data_valid_fold_2.csv'
		)
	tkc_2.make_validation_token_classification()
	'''
	
	'''
	print(' ')

	print(color.BLUE + '--------------- (NER) EVAL CROSS-VALIDATION FOLDER: 3 ---------------' + color.END)

	tkc_3 = EvaluatesTokenClassification(
		path_model_token='/content/drive/My Drive/Colab Notebooks/Fall-Recognition/Experiments/CROSS-NER/FOLDER_3/TRN-FD3/best-model.pt',
		path_valid_data='/content/drive/My Drive/Colab Notebooks/Fall-Recognition/Data/Fall_Extra_Validation.csv',
		output_file_name='tkc_data_valid_fold_3.csv'
		)
	tkc_3.make_validation_token_classification()
	'''

	'''
	print(color.BLUE + '--------------- (TXC) EVAL CROSS-VALIDATION FOLDER: 1 ---------------' + color.END)
	
	txc_1 = EvaluatesTextClassification(
		path_model_text='/content/drive/My Drive/Colab Notebooks/Fall-Recognition/Experiments/CROSS-TXC/FOLDER_1/best-model.pt',
		path_valid_data='/content/drive/My Drive/Colab Notebooks/Fall-Recognition/Data/Fall_Extra_Validation.csv',
		output_file_name='stc_data_valid_fold_1.csv'
		)
	txc_1.make_validation_text_classification()

	print(' ')
	'''

	'''
	print(color.BLUE + '--------------- (TXC) EVAL CROSS-VALIDATION FOLDER: 2 ---------------' + color.END)
	
	txc_2 = EvaluatesTextClassification(
		path_model_text='/content/drive/My Drive/Colab Notebooks/Fall-Recognition/Experiments/CROSS-TXC/FOLDER_2/best-model.pt',
		path_valid_data='/content/drive/My Drive/Colab Notebooks/Fall-Recognition/Data/Fall_Extra_Validation.csv',
		output_file_name='stc_data_valid_fold_2.csv'
		)
	txc_2.make_validation_text_classification()

	print(' ')
	'''

	'''
	print(color.BLUE + '--------------- (TXC) EVAL CROSS-VALIDATION FOLDER: 3 ---------------' + color.END)
	
	txc_3 = EvaluatesTextClassification(
		path_model_text='/content/drive/My Drive/Colab Notebooks/Fall-Recognition/Experiments/CROSS-TXC/FOLDER_3/best-model.pt',
		path_valid_data='/content/drive/My Drive/Colab Notebooks/Fall-Recognition/Data/Fall_Extra_Validation.csv',
		output_file_name='stc_data_valid_fold_3.csv'
		)
	txc_3.make_validation_text_classification()
	'''
	