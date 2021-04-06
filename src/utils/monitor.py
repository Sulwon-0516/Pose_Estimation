import visdom
import torch



''' ref : https://github.com/noagarcia/visdom-tutorial with some modification'''
class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        try:
            self.viz = visdom.Visdom()
        except ConnectionRefusedError:
            print("please turn on Visdom server.\nuse python -m visdom.server")
            assert(0)
        self.env = env_name
        self.figs = {}
        
    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.figs:
            self.figs[var_name] = self.viz.line(
                        X=x, 
                        Y=y, 
                        env=self.env, 
                        opts=dict(
                            legend=[split_name],
                            title=title_name,
                            xlabel='Steps',
                            ylabel=var_name
                        ))
        else:
            self.viz.line(
                        X=x, 
                        Y=y, 
                        env=self.env, 
                        win=self.figs[var_name], 
                        name=split_name, 
                        update = 'append')


