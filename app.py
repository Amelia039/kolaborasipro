import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import pickle5 as pickle 

st.set_page_config(
    page_title="Electric Production Prediction",
    page_icon='https://lh3.googleusercontent.com/DjXftkycmVpJNlxoq-hN1d1-bQ7UkcEr-FeSn6bcFCnnFZb8Y2R0srkfgVqa-GDmlfsfqGxLoFb5o-ukFG_xzq8KSUSBX_369siM0d595.jpg',
    layout='centered',
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)
st.write("""<h1>Aplikasi Prediksi data Time Series pada Dataset Electric Production Electric Production</h1>""", unsafe_allow_html=True)

with st.container():
    with st.sidebar:
        selected = option_menu(
            st.write("""<h2 style = "text-align: center;"><img src="https://lh3.googleusercontent.com/DjXftkycmVpJNlxoq-hN1d1-bQ7UkcEr-FeSn6bcFCnnFZb8Y2R0srkfgVqa-GDmlfsfqGxLoFb5o-ukFG_xzq8KSUSBX_369siM0d59" width="130" height="130"><br></h2>""", unsafe_allow_html=True),
        ["Home", "Description", "Dataset", "Prepocessing", "Modeling", "Implementation"], 
            icons=['house', 'file-earmark-font', 'bar-chart', 'gear', 'arrow-down-square', 'check2-square'], menu_icon="cast", default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#005980"},
                "icon": {"color": "white", "font-size": "18px"}, 
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "color":"white"},
                "nav-link-selected":{"background-color": "#005980"}
            }
        )

    if selected == "Home":
        st.write("""<h3 style = "text-align: center;">
        <img src="data:image/jpeg;base64,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" width="500" height="300">
        </h3>""",unsafe_allow_html=True)

    elif selected == "Description":
        st.subheader("""Pengertian""")
        st.write("""
        Dataset ini merupakan data jumlah produksi listrik setiap dua belas hari per-tahunnya hitungan setiap bulan januari dari tahun 1985 hingga 2018.
        """)

        st.subheader("""Kegunaan Dataset""")
        st.write("""
        Dataset ini digunakan untuk melakukan prediksi perkiraan jumlah produksi listrik.
        """)

        st.subheader("""Fitur""")
        st.markdown(
            """
            Fitur-fitur yang terdapat pada dataset:
            - Period: 
                - Fitur ini merupakan fitur yang perlihatkan periode dari data time series tersebut. Pada fitur ini data tercatat setiap dua belas hari per-tahunnya hitungan setiap bulan januari dari tahun 1985 hingga 2018.
            - IPG2211A2N
                - Fitur berikut ini merupakan data jumlah jumlah produksi listrik.
            """
        )

        st.subheader("""Sumber Dataset""")
        st.write("""
        Sumber data di dapatkan melalui website Github.com, Berikut merupakan link untuk mengakses sumber dataset.
        <a href="https://finance.yahoo.com/quote/BYAN.JK/history/">Klik disini</a>""", unsafe_allow_html=True)
        
        st.subheader("""Tipe Data""")
        st.write("""
        Tipe data yang di gunakan pada dataset yang diambil berupa jumlah produksi listrik.
        """)

    elif selected == "Dataset":
        st.subheader("""Dataset Import Electric_Production""")
        df = pd.read_csv(
            'https://raw.githubusercontent.com/Amelia039/kolaborasipro/main/BYAN.JK.csv')
        st.dataframe(df, width=600)

    elif selected == "Prepocessing":
        st.subheader("""Univariate Transform""")
        uni = pd.read_csv('unvariate4fitur.csv')
        uni = uni.iloc[:, 1:9]
        st.dataframe(uni)
        st.subheader("""Normalisasi Data""")
        st.write("""Rumus Normalisasi Data :""")
        st.image('https://i.stack.imgur.com/EuitP.png', use_column_width=False, width=250)
        df = pd.read_csv('https://raw.githubusercontent.com/HanifSantoso05/dataset_matkul/main/anemia.csv')
        st.markdown("""
        Dimana :
        - X = data yang akan dinormalisasi atau data asli
        - min = nilai minimum semua data asli
        - max = nilai maksimum semua data asli
        """)

        scaler = MinMaxScaler()
        #scaler.fit(features)
        #scaler.transform(features)
        scaledX = scaler.fit_transform(uni)
        features_namesX = uni.columns.copy()
        #features_names.remove('label')
        scaled_featuresX = pd.DataFrame(scaledX, columns=features_namesX)

        st.subheader('Hasil Normalisasi Data')
        st.dataframe(scaled_featuresX.iloc[:,0:7], width=600)

    elif selected == "Modeling":

        uni = pd.read_csv('unvariate4fitur.csv')
        uni = uni.iloc[:, 1:9]

        scaler = MinMaxScaler()
        #scaler.fit(features)
        #scaler.transform(features)
        scaledX = scaler.fit_transform(uni)
        features_namesX = uni.columns.copy()
        #features_names.remove('label')
        scaled_featuresX = pd.DataFrame(scaledX, columns=features_namesX)

        #Split Data 
        training, test = train_test_split(scaled_featuresX.iloc[:,0:7],test_size=0.1, random_state=0,shuffle=False)#Nilai X training dan Nilai X testing
        training_label, test_label = train_test_split(scaled_featuresX.iloc[:,-1], test_size=0.1, random_state=0,shuffle=False)#Nilai Y training dan Nilai Y testing


        st.write("#### Percobaan Model")
        st.markdown("""
        Dimana :
        - Jumlah Fitur Transform Univariet = [1,2,3,4,5] 
        - K = [3,5,7,9,11]
        - Test Size = [0.2,0.3,0.4]
        """)
        df_percobaan = pd.read_csv('hasil_percobaan1.csv')
        st.write('##### Hasil :')
        data = pd.DataFrame(df_percobaan.iloc[:,1:6])
        st.write(data)
        st.write('##### Grafik Pencarian Nilai Error Terkecil :')
        st.line_chart(data=data[['Nilai Error MSE','Nilai Error MAPE']], width=0, height=0, use_container_width=True)
        st.write('##### Best Model :')
        st.info("Jumlah Fitur = 7, K = 3, Test_Size = 0.1, Nilai Erorr MSE= 0.0133, Nilai Error MAPE = 0,085")
        st.write('##### Model KNN :')

        # load saved model
        with open('model_knn_pkl' , 'rb') as f:
            model = pickle.load(f)
        regresor = model.fit(training, training_label)
        st.info(regresor)

            

    elif selected == "Implementation":
        with st.form("Implementation"):
            uni = pd.read_csv('unvariate4fitur.csv')
            uni = uni.iloc[:, 1:9]

            scaler = MinMaxScaler()
            #scaler.fit(features)
            #scaler.transform(features)
            scaledX = scaler.fit_transform(uni)
            features_namesX = uni.columns.copy()
            #features_names.remove('label')
            scaled_featuresX = pd.DataFrame(scaledX, columns=features_namesX)

            #Split Data 
            training, test = train_test_split(scaled_featuresX.iloc[:,0:7],test_size=0.1, random_state=0,shuffle=False)#Nilai X training dan Nilai X testing
            training_label, test_label = train_test_split(scaled_featuresX.iloc[:,-1], test_size=0.1, random_state=0,shuffle=False)#Nilai Y training dan Nilai Y testing

            #Modeling
            # load saved model
            with open('model_knn_pkl' , 'rb') as f:
                model = pickle.load(f)
            regresor = model.fit(training, training_label)
            pred_test = regresor.predict(test)
            
            #denomalize data test dan predict
            hasil_denormalized_test = []
            for i in range(len(test)):
                df_min = uni.iloc[:,0:7].min()
                df_max = uni.iloc[:,0:7].max()
                denormalized_data_test_list = (test.iloc[i]*(df_max - df_min) + df_min).map('{:.1f}'.format)[0]
                hasil_denormalized_test.append(denormalized_data_test_list)

            hasil_denormalized_predict = []
            for y in range(len(pred_test)):
                df_min = uni.iloc[:,0:7].min()
                df_max = uni.iloc[:,0:7].max()
                denormalized_data_predict_list = (pred_test[y]*(df_max - df_min) + df_min).map('{:.1f}'.format)[0]
                hasil_denormalized_predict.append(denormalized_data_predict_list)

            denormalized_data_test = pd.DataFrame(hasil_denormalized_test,columns=["Testing Data"])
            denormalized_data_preds = pd.DataFrame(hasil_denormalized_predict,columns=["Predict Data"])

            #Perhitungan nilai error
            MSE = mean_squared_error(test_label,pred_test)
            MAPE = mean_absolute_percentage_error(denormalized_data_test,denormalized_data_preds)

            # st.subheader("Implementasi Prediksi ")
            v1 = st.number_input('Masukkan Jumlah barrel yang telah di import pada 7 bulan sebelum periode yang akan di prediksi')
            v2 = st.number_input('Masukkan Jumlah barrel yang telah di import pada 6 bulan sebelum periode yang akan di prediksi')
            v3 = st.number_input('Masukkan Jumlah barrel yang telah di import pada 5 bulan sebelum periode yang akan di prediksi')
            v4 = st.number_input('Masukkan Jumlah barrel yang telah di import pada 4 bulan sebelum periode yang akan di prediksi')
            v5 = st.number_input('Masukkan Jumlah barrel yang telah di import pada 3 bulan sebelum periode yang akan di prediksi')
            v6 = st.number_input('Masukkan Jumlah barrel yang telah di import pada 2 bulan sebelum periode yang akan di prediksi')
            v7 = st.number_input('Masukkan Jumlah barrel yang telah di import pada 1 bulan sebelum periode yang akan di prediksi')

            prediksi = st.form_submit_button("Submit")
            if prediksi:
                inputs = np.array([
                    v1,
                    v2,
                    v3,
                    v4,
                    v5,
                    v6,
                    v7
                ])
                
                df_min = uni.iloc[:,0:7].min()
                df_max = uni.iloc[:,0:7].max()
                input_norm = ((inputs - df_min) / (df_max - df_min))
                input_norm = np.array(input_norm).reshape(1, -1)

                st.write("#### Normalisasi data Input")
                st.write(input_norm)

                input_pred = regresor.predict(input_norm)

                st.write('#### Hasil Prediksi')
                st.info((input_pred*(df_max - df_min) + df_min).map('{:.1f}'.format)[0])
                st.write('#### Nilai Error')
                st.write("###### MSE :",MSE)
                st.write("###### MAPE :",MAPE)