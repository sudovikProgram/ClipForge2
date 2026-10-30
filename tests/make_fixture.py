"""Генерация тестового «длинного видео»: 3 минуты, 6 сцен, псевдо-речь
(огибающая AM-тон = ритм слов + паузы между фразами) и субтитры-текстовки."""
import subprocess, os

os.makedirs("tests", exist_ok=True)
# Псевдоречь: тон 300Гц, модулированный пульсами ~3 Гц (слова), с длинными тишинами
am = ("sine=frequency=320:duration=180,"
      "tremolo=f=3.1:d=0.95,"          # пульсация = «слова»
      "volume='if(between(mod(t,18),2,14),1,0.02)':eval=frame")  # фразы по 12с с паузами 6с
scenes = []
for i in range(6):
    scenes.append(f"color=c={'0x%02x%02x%02x' % ((i*40)%256,(i*70)%256,(i*110)%256)}:"
                  f"s=1280x720:r=30:d=30,drawtext=text='TAKE {i+1}':x=(w-text_w)/2:"
                  f"y=(h-text_h)/2:fontsize=110:fontcolor=white")
src = ";".join(scenes)
cmd = ["ffmpeg","-y","-hide_banner","-loglevel","error",
       "-filter_complex",
       f"{';'.join(f'[{i}:v]' for i in range(0))}"  # placeholder
       ]
# проще: concat через filter
inputs=[]; fc=[]
for i in range(6):
    inputs += ["-f","lavfi","-i", f"color=c=0x{((i*40)%256):02x}{((i*70)%256):02x}{((i*110)%256):02x}:s=1280x720:r=30:d=30"]
    inputs += ["-f","lavfi","-i", am.replace("180","30")]
fc = "".join(f"[{2*i}:v][{2*i+1}:a]" for i in range(6))
fc += f"concat=n=6:v=1:a=1[v][a]"
# поверх — движущийся белый прямоугольник (имитация движения в кадре)
fc2 = "[v]drawbox=x='mod(t*160,900)':y=200:w=300:h=300:color=white@0.85:t=fill,drawtext=text='TAKE %{eif\\:floor(t/30)+1\\:d}':x=40:y=40:fontsize=64:fontcolor=yellow:box=1[vout]"
cmd = ["ffmpeg","-y","-hide_banner","-loglevel","error"] + inputs + \
      ["-filter_complex", fc + ";" + fc2, "-map","[vout]","-map","[a]",
       "-c:v","libx264","-preset","ultrafast","-pix_fmt","yuv420p",
       "-c:a","aac","tests/long_test.mp4"]
print(subprocess.run(cmd, capture_output=True, text=True).stderr[-500:])
subprocess.run(["ffprobe","-v","error","-show_entries","format=duration",
                "-of","default=nw=1:nk=1","tests/long_test.mp4"])
