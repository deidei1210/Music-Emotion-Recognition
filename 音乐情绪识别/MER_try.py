import numpy as np
import pandas as pd
import librosa
from keras.models import load_model
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pickle
import joblib

model = load_model('MER_model.h5')
model2 = joblib.load('cnn_model.pkl')
# NOISE
def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

# STRETCH
def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(y=data, rate=rate)

# SHIFT
def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)

# PITCH
def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(y=data, sr=sampling_rate, n_steps=pitch_factor)


# Zero crossing rate
def zcr(data,frame_length,hop_length):
    zcr=librosa.feature.zero_crossing_rate(data,frame_length=frame_length,hop_length=hop_length)
    return np.squeeze(zcr)

# Root mean square
def rmse(data,frame_length=2048,hop_length=512):
    rmse=librosa.feature.rms(y=data,frame_length=frame_length,hop_length=hop_length)
    return np.squeeze(rmse)

# Mel-Frequency Cepstral coefficient
def mfcc(data,sr,frame_length=2048,hop_length=512,flatten:bool=True):
    mfcc=librosa.feature.mfcc(y=data,sr=sr)
    return np.squeeze(mfcc.T)if not flatten else np.ravel(mfcc.T)

# Combine all feature functions
def extract_features(data,sr=22050,frame_length=2048,hop_length=512):
    result=np.array([])
    
    result=np.hstack((result,
                      zcr(data,frame_length,hop_length),
                      rmse(data,frame_length,hop_length),
                      mfcc(data,sr,frame_length,hop_length)
                     ))
    return result

# Apply data augmentation and extract its features
def get_features(path,duration=28, offset=0.6):
    data,sr=librosa.load(path,duration=duration,offset=offset,mono=True)
    aud=extract_features(data)
    audio=np.array(aud)
    
    # noised_audio=noise(data)
    # aud2=extract_features(noised_audio)
    # audio=np.vstack((audio,aud2))
    
    # pitched_audio=pitch(data,sr)
    # aud3=extract_features(pitched_audio)
    # audio=np.vstack((audio,aud3))
    
    # pitched_audio1=pitch(data,sr)
    # pitched_noised_audio=noise(pitched_audio1)
    # aud4=extract_features(pitched_noised_audio)
    # audio=np.vstack((audio,aud4))
    
    return audio

def predict_emotion(file_path, model):
    Y = np.load('Y.npy')
    print(Y)
    encoder = OneHotEncoder()
    Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()
    
    # 提取特征
    features=get_features(input_filename)
    # 将特征转换为适合CNN模型输入的形状
    # feature_cnn = np.expand_dims(features, axis=2)
    print(features.shape)
    test=np.array([features.reshape(26532,1)])
    print(test)
    # 使用模型进行预测
    predictions = model2.predict(test)
    # y_pred0 = encoder.inverse_transform(predictions)

    print(test.shape)
    print(predictions)


    # 获取最终预测结果
    predicted_label = np.argmax(predictions, axis=1)
    return predicted_label

if __name__ == "__main__":
    input_filename="/Users/mac/Desktop/作业/大三下/MER相关论文/音乐情绪识别/MIREX/350.mp3"
    # 使用模型进行情感预测
    predicted_emotion = predict_emotion(input_filename, model)
    print(predicted_emotion)

    # X,Y=[],[]
    # features=get_features(input_filename)
    # for i in features:
    #     X.append(i)

    # Emotions = pd.DataFrame(X)
    # X = Emotions.values

    # print("X的形状:")
    # print(X.shape)

    # Y=np.load('Y.npy')
    # encoder = OneHotEncoder()
    # Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()

    # # 从文件中加载标准化器对象
    # with open('scaler.pkl', 'rb') as f:
    #     scaler = pickle.load(f)

    # # 对测试数据进行转换
    # x_test = scaler.transform(X)
    # x_testcnn= np.expand_dims(x_test, axis=2)

    # # 加载模型
    # loaded_model = load_model('MER_model.h5')

    # #检查X的格式以及shape
    # print("x_testcnn的大小:")
    # print(x_testcnn[0].shape)
    # print("x_testcnn的内容:")
    # print(x_testcnn[0])

    

    # # 使用加载的模型进行预测
    # y_pred = loaded_model.predict(x_testcnn[0])
    # y_pred0 = encoder.inverse_transform(y_pred)

    # print(y_pred)

    # print(y_pred0)


