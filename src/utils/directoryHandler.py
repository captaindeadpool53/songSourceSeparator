import os
import shutil

class DirectoryHandler:
	def __init__(self) -> None:
		pass
	
	@staticmethod
	def deleteAllOcurrancesOfDirectory(rootPath, target ):
		for root, folders, files in os.walk(rootPath):
				for folder in folders:
					if(os.path.splitext(folder)[0] == target):
						shutil.rmtree(os.path.join(root, folder))

	"""
	For windows path issue
 	"""
	@staticmethod
	def joinPath(path1, path2):
		path = os.path.join(path1, path2)
		path = path.replace('\\','/')
		return path
# if __name__ == "__main__":
# 	DirectoryAndFileHandler.deleteAllOcurrancesOfDirectory("data/babyslakh_16k","training_data_drums")