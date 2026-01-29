"""
A ThreadPoolExecutor with lazy execution (see documentation of concurrent.futures in Python 3.14).
"""
import concurrent.futures
from typing import override
import time
import logging

logger = logging.getLogger()

class ThreadPoolExecutor(concurrent.futures.ThreadPoolExecutor):
    @override
    def map(self, fn, *iterables, timeout = None, chunksize = 1, buffersize = None, jobdelay: float = 0):
        """
        This overridden function allows lazy execution through the buffersize parameter.
        See map() documentation in concurrent.futures for details.
        jobdelay (in seconds) adds a delay before launching each job, intended to prevent 'spamming' a server with requests.
        """
        if buffersize is None:
            return super().map(fn, *iterables, timeout = timeout, chunksize = chunksize)

        if chunksize > 1:
            logger.warning(f"Chunksize support not implemented in custom ThreadPoolExecutor")

        iterator = zip(*iterables)
        iterator_empty = False

        # Start by filling the buffer
        future_to_index = {}
        next_index = 0
        while len(future_to_index) < buffersize:
            try:
                time.sleep(jobdelay)
                future_to_index[self.submit(fn, *next(iterator))] = next_index
                next_index += 1
            except StopIteration:
                iterator_empty = True
                break

        # Wait for any job to finish, then add a new one if possible
        results = [None,] * len(list(zip(*iterables)))
        while future_to_index:
            # Retrieve new results
            done, _ = concurrent.futures.wait(future_to_index, timeout = timeout, return_when = concurrent.futures.FIRST_COMPLETED)

            # Process new results
            for future in done:
                index = future_to_index[future]
                results[index] = future.result(timeout = timeout)
                del future_to_index[future]

            # Add new job if possible
            if not iterator_empty:
                try:
                    time.sleep(jobdelay)
                    future_to_index[self.submit(fn, *next(iterator))] = next_index
                    next_index += 1
                except StopIteration:
                    iterator_empty = True
                    
        return results