import subprocess
import os
import threading
env=os.environ
lock =threading.Lock()
lock.acquire()

env['LC_ALL'] = 'en_US.UTF_8'
METEOR_JAR='meteor-1.5.jar'

meteor_cmd = ['java', '-jar', '-Xmx2G', METEOR_JAR, \
                   '-', '-', '-stdio', '-l', 'en', '-norm']
meteor_p = subprocess.Popen(meteor_cmd, \
                                 cwd=os.path.dirname(os.path.abspath(__file__)), \
                                 stdin=subprocess.PIPE, \
                                 stdout=subprocess.PIPE, \
                                 stderr=subprocess.PIPE,
                                 env=env, universal_newlines=True, bufsize=1)
def _stat(hypothesis_str, reference_list):
    # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
    hypothesis_str = hypothesis_str.replace('|||', '').replace('  ', ' ')
    score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
    meteor_p.stdin.write(score_line + '\n')
    return meteor_p.stdout.readline().strip()
o=_stat('i like',['i','love'])
eval_line = 'EVAL'
eval_line += ' ||| {}'.format(o)
meteor_p.stdin.write(eval_line + '\n')
socre=meteor_p.stdout.readline().strip()
print(socre)
meteor_p.stdin.flush()
lock.release()