import logging
import owncloud
import os
import tempfile
import cv2

logger = logging.getLogger(__name__)


class Uploader:
    def __init__(
        self,
        password,
        username="hand-xray.pbox@uni-bonn.de",
        url="https://uni-bonn.sciebo.de",
        path_in_cloud="bone_age_deployment/input/",
    ):
        self.ow = owncloud.Client(url)
        self.ow.login(username, password)
        self.path = path_in_cloud

    def upload_image(self, img, name):
        loc_path = os.path.join(tempfile.gettempdir(), name)
        cv2.imwrite(loc_path, img)
        self.ow.put_file(os.path.join(self.path, name), loc_path)
        logger.info(f"uploaded file {name}")
        os.remove(loc_path)

    def __call__(self, img, name):
        self.upload_image(img, name)
