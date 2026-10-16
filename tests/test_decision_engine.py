import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from decision_engine import DriverMonitor

class Point:
    def __init__(self,x,y): self.x=x; self.y=y
class Shape:
    def __init__(self,category_name,score):
        self.category_name=category_name; self.score=score

def face(mouth=0.01, turn=0.0):
    pts = [Point(.5, .5) for _ in range(478)]
    pts[33]=Point(.35,.40); pts[263]=Point(.65,.40)
    pts[1]=Point(.5+turn,.52); pts[152]=Point(.5,.80)
    pts[61]=Point(.42,.68); pts[291]=Point(.58,.68)
    pts[13]=Point(.5,.67-mouth/2); pts[14]=Point(.5,.67+mouth/2)
    return pts

def test_no_face_unknown():
    m=DriverMonitor()
    assert m.analyze(None,[],0)['state']=='NO_FACE'
    assert not m.analyze(None,[],100)['alert']

def test_calibration_forward():
    m=DriverMonitor()
    for i in range(14): out=m.analyze(face(),[Shape('jawOpen',.0)],i*200)
    assert out['calibrated'] and out['state']=='FOCUSED'

def test_consecutive_head_turn_alert():
    m=DriverMonitor()
    for i in range(12): m.analyze(face(),[],i*200)
    for i in range(12,26): out=m.analyze(face(turn=.14),[],i*200)
    assert out['head_away'] and out['alert'] and out['episodes']==1

def test_talking_and_away_alert():
    m=DriverMonitor()
    for i in range(12): m.analyze(face(),[],i*200)
    for i in range(12,42):
        size=.005 if i%2==0 else .11
        out=m.analyze(face(mouth=size,turn=.14),[Shape('jawOpen',.05 if i%2==0 else .7)],i*200)
    assert out['speaking'] and out['head_away'] and out['alert']
    assert out['state']=='ALERT_TALK_AWAY'

if __name__=='__main__':
    test_no_face_unknown(); test_calibration_forward(); test_consecutive_head_turn_alert(); test_talking_and_away_alert(); print('4 tests passed')
