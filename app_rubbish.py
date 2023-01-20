import lightning as L
import time


class Work(L.LightningWork):
    def run(self):
        while True:
            print(f"hello world! I'm {id(self)}")
            time.sleep(1)


class Flow(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.ws = L.app.structures.List()
        self.fake_counter = 0
        self.interval = 5  # sec
        self.start_time = time.time()

    def run(self):
        if self.interval < time.time() - self.start_time:
            work = Work()
            self.ws.append(work)
            print(f"Added a new work {id(work)}")
            self.start_time = time.time()

        for work in self.ws:
            work.run()
        print(self.ws)
        self.fake_counter += 1


app = L.LightningApp(Flow())
