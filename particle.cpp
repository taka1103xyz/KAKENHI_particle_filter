#include <math.h>
#include <fstream>
#include <iostream>
//#include <stdio.h>
#include <random>
#include <Eigen/Core>
//#include <Eigen/Geometry>

using namespace std;
using namespace Eigen;

int
main (int argc, char **argv)
{
        // 事前設定
    char p_data[100];
    char p_figure[100];


      // パラメータ設定
    int A = 1;
    // int b = 1;
    int c = 1;
    int P = 0;
    int Q = 2;                      // システムノイズ
    int R = 2;                      // 観測ノイズ
    // int dt = 1;                  // 時間間隔
    int M = 200;                    // パーティクル数
    int N = 200;                    // 計測数
    double sigma = sqrt(0.05);            // パーティクルフィルタのパラメータ σ^2
    double ESS = 0;                 // リサンプリング閾値
    double ratio = 0;                  // dの値が0に近い粒子がいくつあるか検証（比率がわかる）


        //   randomの定義
    random_device rnd;
    mt19937 mt(rnd());
    normal_distribution<> norm(0.0, sqrt(0.05));        // 標準正規乱数
    normal_distribution<> norm_bias(1.0, sqrt(0.1));    // 標準正規乱数
    uniform_real_distribution<double> score(0.0,1.0);   // 0から1までの小数乱数
    // out << score(mt) << endl;

        // 行列の定義
    MatrixXd x      = MatrixXd::Zero(1,N);           // システム
    MatrixXd y1     = MatrixXd::Zero(1,N);           // 観測値1
    MatrixXd y2     = MatrixXd::Zero(1,N);           // 観測値2
    MatrixXd phat   = MatrixXd::Zero(1,N);           // パーティクルによる推定値
    MatrixXd d      = MatrixXd::Zero(1,M);           // 差分評価
    MatrixXd weight_power = MatrixXd::Zero(1,N);     // リサンプリングするかどうか閾値に使う
    MatrixXd d_mean = MatrixXd::Zero(1,N);           // dの平均値をまとめてみた
    MatrixXd p_mean = MatrixXd::Zero(1,N);           // pの平均値をまとめてみた

        // パーティクル行列
    MatrixXd p    = MatrixXd::Zero(1,M);

        // 重み
    MatrixXd weight = MatrixXd::Constant(1,M,1.0/M);
    //MatrixXd weight_p = MatrixXd::Constant(1,M,1.0/M);  // 観測値をyと置かずにp_meanを観測値とおいた場合を検証

        // datファイルの生成
    VectorXd t = VectorXd::LinSpaced(N,0,N-1);
    ofstream fout("../data/output.dat");
    ofstream dout("../data/d_mean.dat");
    ofstream absdout("../data/absd_mean.dat");

        // 初期値の設定
    y1(0,0) = c * x(0,0) + sqrt(R)*norm(mt);
    y2(0,0) = c * x(0,0) + sqrt(R)*norm_bias(mt);

        //ファイルへデータを出力。
    fout << t(0) << " " << x(0,0) << " "<< y1(0,0) << " " << y1(0,0) << " " << p(0,0) << endl;


        // 観測とパーティクルを使った推定
    for(int k=1;k<N;k++){

            // 保存するための準備
        sprintf(p_data,  "../data/particle%i.dat",k);
        sprintf(p_figure,"../figure/figure%i.png",k);
        ofstream pout(p_data);
        FILE *gp = popen("gnuplot -persist", "w");; // For gnuplot
        if (gp == NULL){
            return -1;
        }
        cout << k << " Loop" << endl;

            // 時間更新
        x(0,k) = A * x(0,k-1) + sqrt(Q)*norm(mt);   // システム更新
        ////////////////////////////////////////////////////////////////////////////////////
        //////////////////////                観測値             ////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////
            // 正常データ
        //y1(0,k) = c * x(0,k) + sqrt(R)*norm(mt);    // 観測値更新

            // 故障データ（周期ごとに同じ値を得る）
        /*
        if(k > 25){
            if(k % 5 == 0) {
                y1(0, k) = x(0,25);
            }else{
                y1(0,k) = c * x(0,k) + sqrt(R)*norm(mt);    // 観測値更新
            }
        } else{
            y1(0,k) = c * x(0,k) + sqrt(R)*norm(mt);    // 観測値更新
        }
        */
            // 故障データ(ある時刻で同じ値を得る)
        /*
        if(k > 25){
            y1(0,k) = 0; //c * x(0,k) + sqrt(R)*norm(mt) + 5;
        } else{
            y1(0,k) = c * x(0,k) + sqrt(R)*norm(mt);    // 観測値更新
        }
        */
            // 故障データ（事前にセットしていたシステムとは異なる値（バイアス、大きな分散）が得られた場合）

        if(k > 25){
            y1(0,k) = c * x(0,k) + sqrt(30)*norm(mt);
        } else{
            y1(0,k) = c * x(0,k) + sqrt(R)*norm(mt);    // 観測値更新
        }

        // y2(0,k) = c * x(0,k) + sqrt(R)*norm_bias(mt);    // 観測値更新
        // cout << "Update" << endl;
        cout << "x(" << k << ") is " << x(0,k) << endl;
        cout << "y1(" << k << ") is " << y1(0,k) << endl;
        cout << "y2(" << k << ") is " << y2(0,k) << endl;

            // サンプリング（比率の計算をここでやる）
        ratio = 0;
        for(int i=0;i<M;i++){
            p(0,i) = A * p(0,i) + sqrt(Q)*norm(mt);
            d(0,i) = p(0,i) - y1(0,k);
            if(abs(d(0,i))<0.1){
                ratio++;
            }
            pout << d(0,i) << endl;
        }
        cout << "ratio is " << ratio/M << endl;

            // 平均値の算出
        d_mean(0,k) = d.sum()/M;
        p_mean(0,k) = p.sum()/M;




        //cout << "Sampling" << endl;
        //cout << p << endl;

            // 尤度の計算
        for(int i=0;i<M;i++){
            weight(0,i) = exp(-pow((p(0,i)-y1(0,k)),2.0)/(2.0*pow(sigma,2.0)));
            //weight_p(0,i) = exp(-pow((p(0,i)-p_mean(0,i)),2.0)/(2.0*pow(sigma,2.0)));
            /*
                // 2つのセンサを使った観測値の算出
            weight(0,i) = exp(-pow((p(0,i)-y1(0,k)),2.0)/(2.0*pow(sigma,2.0))) + exp(-pow((p(0,i)-y2(0,k)),2.0)/(2.0*pow(sigma,2.0)));
            */
        }

            // 重みの正規化
        double weight_sum = weight.sum();
        weight = weight / weight_sum;

        /*
        cout << "Weight" << endl;
        cout << weight << endl;
        */

            // 重みの累積和の算出
        for(int i=1;i<M;i++){
            weight(0,i) = weight(0,i) + weight(0,i-1);
        }
        /*
        cout << "累積和" << endl;
        cout << weight << endl;
        */
    
            // リサンプリング「ランダムサンプリング」
            // 分散がある閾値を超えるまではリサンプリングを行わない
        weight_power = weight.array()*weight.array();
        ESS = 1/weight_power.sum();
        cout << "Dispersion is " << weight_power.sum()/M << endl;
        cout << "ESS = " << ESS << endl;

        if(0 == 0){
            MatrixXd p_bar = MatrixXd::Zero(1,M);
            for(int i=0;i<M;i++){
                double sample = score(mt);
                int num = 0;
                for(int j=0;j<M;j++){
                    if((weight(0,j)<sample)&&(sample<weight(0,j+1))){
                        num = j+1;
                    }
                }
                p_bar(0,i) = p(0,num);
            }
        p = p_bar;
        }
        /*
        cout << "Resampling" << endl;
        cout << p << endl;
        */
    
            // 重み付き平均の算出
        phat(0,k) = p.sum() / M;
        /*
        cout << "weighted mean" << endl;
        cout << phat(0,k) << endl << endl;
        */
        cout << "phat(" << k << ") is " << phat(0,k) << endl << endl;

            //ファイルへデータを出力。
        fout << t(k) << " " << x(0,k) << " "<< y1(0,k) << " " << y2(0,k) << " " << p(0,k) << endl;  // t x y p
        dout << k << " " << d_mean(0,k) << " " << ratio/M*10 << " " << weight_power.sum()/M*10 << endl;   // 平均　比率　分散

            // ヒストグラムの生成
        fprintf(gp, "binwidth=0.1\n");
        fprintf(gp, "set title 'particle hist'\n");
        fprintf(gp, "bin(x,width)=width*floor(x/width)+width/2.0\n");
        fprintf(gp, "set terminal png\n");
        fprintf(gp, "set output \"%s\" \n", p_figure);
        fprintf(gp, "plot [-5:5] [0:140] \"%s\" using (bin($1,binwidth)):(1.0) smooth freq with boxes\n", p_data);
        //fprintf(gp, "set terminal png\n");
        //fprintf(gp, "set out \"%s\" \n", p_figure);
        //fprintf(gp, "replot\n");
        pclose(gp);
    }
        //---Gnuplotのコマンドを実行---
    char* output_data = (char*)"../data/output.dat";
    char* output_d = (char*)"../data/d_mean.dat";
    char* output_figure_d = (char*)"../figure/result.png";
    char* output_figure_a = (char*)"../figure/analysis.png";

        // 結果データ
    FILE *gp = popen("gnuplot -persist", "w");; // For gnuplot
    if (gp == NULL){
        return -1;
    }
    cout << "finish" << endl;
  
        // 座標の名前を入力
    fprintf(gp, "set xlabel \"t\"\n");
    fprintf(gp, "set ylabel \"x\"\n");
    fprintf(gp, "set title 'Result data'\n");
    fprintf(gp, "plot \"%s\" using 1:2 title 'x' with lines \n", output_data);     // システム
    fprintf(gp, "replot \"%s\" using 1:3 title 'y1' with lines \n", output_data);  // 観測
    //fprintf(gp, "replot \"%s\" using 1:4 title 'y2' with linespoints \n", output);  // 観測
    fprintf(gp, "replot \"%s\" using 1:5 title 'phat' with lines \n", output_data);   // パーティクル推定
    fprintf(gp, "set terminal png\n");
    fprintf(gp, "set out \"%s\" \n", output_figure_d);
    fprintf(gp, "replot\n");
    pclose(gp);

        // データ分析
    FILE *dp = popen("gnuplot -persist", "w");; // For gnuplot
    if (dp == NULL){
        return -1;
    }

    fprintf(dp, "set xlabel \"t\"\n");
    fprintf(dp, "set ylabel \"x\"\n");
    fprintf(gp, "set title 'Data analysis'\n");
    fprintf(dp, "plot [0:\"%i\"] [-5:5] \"%s\" using 1:2 title 'mean' with linespoints \n",N , output_d);   // 平均
    fprintf(gp, "replot \"%s\" using 1:3 title 'ratio(*10)' with linespoints \n", output_d);   // パーティクル推定
    fprintf(gp, "replot \"%s\" using 1:4 title 'dispersion(*10)' with linespoints \n", output_d);   // パーティクル推定
    fprintf(gp, "set terminal png\n");
    fprintf(gp, "set out \"%s\" \n", output_figure_a);
    fprintf(dp, "replot\n");
    pclose(dp);

    return 0;
}
