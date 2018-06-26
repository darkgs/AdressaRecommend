
import os

def get_files_under_path(p_path=None):
	ret = []

	if p_path == None:
		return ret

	for r, d, files in os.walk(p_path):
		for f in files:
			file_path = os.path.join(r,f)
			if not os.path.isfile(file_path):
				continue

			ret.append(file_path)

	return ret
