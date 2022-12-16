from pathlib import Path
from typing import Optional
import subprocess

from lightning.app import LightningFlow, LightningApp, LightningWork


class LocustWork(LightningWork):
    def __init__(self, work_id: int, is_master: Optional[bool] = False):
        super().__init__(parallel=True)
        self.work_id = work_id
        self.is_master = is_master

    def run(self, master_ip: Optional[str] = None, master_port: Optional[int] = None):

        if not self.is_master and (master_ip is None or master_port is None):
            raise ValueError("master_ip is required for slave nodes")

        if self.is_master:
            command = ["locust", "--web-host", "0.0.0.0", "--web-port", str(self.port), "--master"]
        else:
            command = ["locust", "--worker", "--master-host", master_ip]

        subprocess.run(command, cwd=Path(__file__).parent, check=True)


class Root(LightningFlow):
    def __init__(self):
        super().__init__()
        self.master = LocustWork(0, is_master=True)
        self.slave1 = LocustWork(1)

    def run(self):
        self.master.run()
        if self.master.is_running and self.master.internal_ip:
            self.slave1.run(master_ip=self.master.internal_ip, master_port=self.master.port)

    def configure_layout(self):
        return {"name": "Dashboard", "content": self.master.url}


app = LightningApp(Root())
