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


# if __name__ == "__main__":
# 	DirectoryAndFileHandler.deleteAllOcurrancesOfDirectory("data/babyslakh_16k","training_data_drums")