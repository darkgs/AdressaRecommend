
import os, sys
import signal
import time

from threading import Thread
from threading import Semaphore

from ad_util import write_log

class MultiWorker(object):
	def __init__(self, worker_count=5):
		self._time_to_die = False
		self._worker_count = worker_count

		signal.signal(signal.SIGINT, self.gracefull_die)
		signal.signal(signal.SIGTERM, self.gracefull_die)

#	def __del__(self):
#		print ('distructor')

	def work(self, works, work_function):
		self._working_sema = Semaphore(1)
		self._child_count = 0

		total_work_count = len(works)
		cur_work_done = 0

		for work in works:
			while(True):
				if (self._child_count < self._worker_count):
					break
				time.sleep(1)

			cur_work_done += 1
			if ((cur_work_done % 1000) == 0):
				write_log('working : {}/{}'.format(cur_work_done, total_work_count))

			if (self._time_to_die):
				break

			def run_on_subproc(work):
				child_pid = os.fork()
				# child process
				if (child_pid == 0):
					work_function(work)
					exit(0)
				os.waitpid(child_pid, 0)

				self._working_sema.acquire()
				self._child_count -= 1
				self._working_sema.release()

			self._working_sema.acquire()
			self._child_count += 1
			Thread(target=run_on_subproc, args=work).start()
			self._working_sema.release()

		while(self._child_count > 0):
			time.sleep(1)


	def gracefull_die(self, signum, frame):
		self._time_to_die = True
		print ('signal recieved.. waiting threads die')


def main():
	multi_worker = MultiWorker(worker_count=5)

	def print_work(work):
		print(str(work))

	works = map(lambda x:tuple([x]), range(100))
	multi_worker.work(works=works, work_function=print_work)

	multi_worker = None

if __name__ == '__main__':
	main()

