import pandas as pd
import joblib 


# ------------------------------------------
#  predict 
# ------------------------------------------
def do_predict(model_file,test_file,ans_file):
    df = pd.read_csv(test_file)

    x = df.values 

    model = joblib.load(model_file)

    y_pre = model.predict(x)
    y_pre = y_pre.reshape(-1,1)

    df1 = pd.DataFrame(y_pre)
    df1.columns = ['class']

    df1.to_csv(ans_file,index=False)

# 주의) 위의 코드는 변경이 필요없고, 아래의 값만 변경하면 됩니다.
# 저장된 모델 파일, 문제파일, 시험답안 파일(이름1)
do_predict('../m1.m','test_2.csv','홍길동2.csv')

 

