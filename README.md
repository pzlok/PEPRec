<br>

**项目结构**
```
transelate
    ├── exercise.csv           运动项目
    ├── marge_data.csv         整合后的数据集（韩语未翻译）
    ├── README.md              说明文档
    ├── tran.py                翻译的代码
    ├── tran_data_EN.csv       翻译成英文后的数据集
    ├── tran_data_ZH.csv       翻译成中文后的数据集
```

**数据集中的字段说明**
```
age class                                        年龄分类
age                                              年龄
score class                                      体质测试成绩等级
gender                                           性别
height_cm                                        身高
weight_kg                                        体重
body fat_%                                       体脂
waistline_cm                                     腰围
diastolic_mmHg                                   舒张压
systolic_mmHg                                    收缩压
left grip_kg                                     左握力
right grip_kg                                    右握力
sit-ups_times                                    仰卧起坐
side jump_times                                  重复侧身跳跃
sit and bend forward_cm                          坐体前屈
illinois agility test_s                          伊利诺伊敏捷性检查
teenagers hang time_s                            滞空时间
coordination time_s                              协调时间
coordination errors_times                        协调失误次数
coordination value                               协调计算结果值
BMI_kg/㎡                                        身体质量指数
cross sit-ups_times                              交叉仰卧起坐
turn-back run_times                              往返跑
10m turn-back run_s                              10M 4次往返跑
standing broad jump_cm                           立定跳远
30s chair stand test_times                       30秒坐站测试
6min walk_m                                      步行6分钟
2min standing step_times                         2分钟步行
sit and return_s                                 坐在椅子上返回3米目标
8 word walk_s                                    8字行走
relative grip_%                                  相对握力
turn-back run oxygen uptake_VO₂max               往返长跑最大摄氧量
stable treadmill_bpm                             跑步机_稳定时
3min treadmill_bpm                               跑步机_3分钟
6min treadmill_bpm                               跑步机_6分钟
9min treadmill_bpm                               跑步机_9分钟
treadmill output_VO₂max                          跑步机输出
step test heart rate_bpm                         台阶测试_恢复时的心率
step test oxygen uptake_VO₂max                   台阶测试最大摄氧量
reaction time_s                                  反应时
adult hang time_s                                滞空时
exercise prescription                            运动处方（只包含运动项）
```

**数据来源**
```
数据集来源网址
https://www.bigdata-culture.kr/bigdata/user/data_market/detail.do?id=b3924850-aa65-11ec-8ee4-95f65f846b27

体质测试项目视频网址
https://nfa.kspo.or.kr/reserve/0/selectMeasureItemListByAgeSe.kspo

运动处方项目视频网址
https://nfa.kspo.or.kr/classroom/program/selectPrescriptionMovieList.kspo
```