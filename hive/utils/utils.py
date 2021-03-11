import os


def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


class Chomp:
    def __init__(self):
        self.__dict__["tparams"] = OrderedDict()

    def __setattr__(self, name, array):
        tparams = self.__dict__["tparams"]
        tparams[name] = array

    def __setitem__(self, name, array):
        self.__setattr__(name, array)

    def __getitem__(self, name):
        return self.__getattr__(name)

    def __getattr__(self, name):
        tparams = self.__dict__["tparams"]
        if name in tparams:
            return tparams[name]
        else:
            return None

    def remove(self, name):
        del self.__dict__["tparams"][name]

    def get(self):
        return self.__dict__["tparams"]

    def values(self):
        tparams = self.__dict__["tparams"]
        return list(tparams.values())

    def save(self, filename):
        tparams = self.__dict__["tparams"]
        pickle.dump({p: tparams[p] for p in tparams}, open(filename, "wb"), 2)

    def load(self, filename):
        tparams = self.__dict__["tparams"]
        loaded = pickle.load(open(filename, "rb"))
        for k in loaded:
            tparams[k] = loaded[k]

    def setvalues(self, values):
        tparams = self.__dict__["tparams"]
        for p, v in zip(tparams, values):
            tparams[p] = v

    def add_from_dict(self, src_dict):
        for key in src_dict.keys():
            self.__setattr__(key, src_dict[key])

    def __enter__(self):
        _, _, _, env_locals = inspect.getargvalues(inspect.currentframe().f_back)
        self.__dict__["_env_locals"] = list(env_locals.keys())

    def __exit__(self, type, value, traceback):
        _, _, _, env_locals = inspect.getargvalues(inspect.currentframe().f_back)
        prev_env_locals = self.__dict__["_env_locals"]
        del self.__dict__["_env_locals"]
        for k in list(env_locals.keys()):
            if k not in prev_env_locals:
                self.__setattr__(k, env_locals[k])
                env_locals[k] = self.__getattr__(k)
        return True

    def __deepcopy__(self, memo):
        new_container = Chomp()
        new_container.add_from_dict(src_dict=deepcopy(self.get()))
        return new_container
