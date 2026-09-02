from PIL import Image
import rembg

class BackgroundRemover():
    _session = None

    @classmethod
    def get_session(cls):
        if cls._session is None:
            try:
                cls._session = rembg.new_session("birefnet-general", providers=["CPUExecutionProvider"])
            except Exception:
                try:
                    cls._session = rembg.new_session("u2net", providers=["CPUExecutionProvider"])
                except Exception:
                    cls._session = False
        return cls._session if cls._session is not False else None

    def __init__(self):
        pass

    def __call__(self, image: Image.Image):
        sess = self.get_session()
        if sess is not None:
            return rembg.remove(image, session=sess)
        return rembg.remove(image)
