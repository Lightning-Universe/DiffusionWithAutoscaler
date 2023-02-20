import subprocess
from pathlib import Path
from typing import Optional

from lightning.app import LightningApp, LightningFlow, LightningWork


class LocustWork(LightningWork):
    def __init__(self, work_id: int, is_master: Optional[bool] = False):
        super().__init__(parallel=True)
        self.work_id = work_id
        self.is_master = is_master

    def run(self, master_host: Optional[str] = None, master_port: Optional[int] = None):
        if not self.is_master and (master_host is None or master_port is None):
            raise ValueError("master_host is required")

        if self.is_master:
            command = ["locust", "--web-host", "0.0.0.0", "--web-port", str(self.port), "--master"]
        else:
            command = ["locust", "--worker", "--master-host", master_host]

        subprocess.run(command, cwd=Path(__file__).parent, check=True)


class Root(LightningFlow):
    def __init__(self):
        super().__init__()
        self.master = LocustWork(0, is_master=True)
        self.slave1 = LocustWork(1)

    def run(self):
        self.master.run()
        if self.master.is_running and self.master.internal_ip:
            self.slave1.run(master_host=self.master.internal_ip, master_port=self.master.port)

    def configure_layout(self):
        return {"name": "Dashboard", "content": self.master.url}


app = LightningApp(Root())
