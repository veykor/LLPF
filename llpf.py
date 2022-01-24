'''LLPF or ListLowPassFilter is a low pass filter for list type, can use per example for filter to neural network vector out.
All right reserved to veykor(https://github.com/veykor)
Version:1.0
'''

from vcommon import *

class ListLowPassFilter:
	shape = 0
	filterBuffer = []
	filterSize = 0
	
	def __init__(self,inputShape,filterSize): 
		logger(0,str(inputShape)+' - '+str(filterSize))
		
		self.shape = inputShape
		self.filterBuffer = [[0.]*filterSize]*inputShape
		self.filterSize = filterSize

	def sumList(self,listIn):
		res = 0
		for l in listIn:
			res += l
		return res

	def checkShape(self,inputs):
		return (self.shape == len(inputs))

	def initFilterBuffer(self,initValues):
		if self.checkShape(initValues) == False:
			logger(2,'Shape init values and shape filter not equal in initFilterBuffer')
			return
		for i in range(self.shape):
			self.filterBuffer[i] = [initValues[i]]*self.filterSize

	def step(self,rInputs):
		inputs = list(rInputs)
		if self.checkShape(inputs) == False:
			logger(3,'Shape input and shape filter not equal in step')
			exit(1)
		res = [0]*self.shape
		for i in range(self.shape):
			self.filterBuffer[i] = self.filterBuffer[i][1:]
			self.filterBuffer[i].append(inputs[i])
			res[i] = self.sumList(self.filterBuffer[i])/self.filterSize
		return res