class DataPoint:
	def __init__(self, image, num_of_cubes):
		self.image = image
		self.num_of_cubes = num_of_cubes
	def setNumberOfCubes(num_of_cubes):
		self.num_of_cubes=num_of_cubes
	def getImage(self):
		return self.image



class TestPoint:
	def __init__(self, image, ground_truth, prediction):
		self.image = image
		self.prediction = prediction
		self.ground_truth = ground_truth
	
	def setImage(image):
		self.image = image
	
	def setGroundTruth(ground_truth):
		self.ground_truth=ground_truth
	
	def setGroundTruth(prediction):
		self.prediction=prediction
	
	def getImage(self):
		return self.image

	def getGroundTruth(self):
		return self.ground_truth

	def getPrediction(self):
		return self.prediction