#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <eigen3/unsupported/Eigen/MatrixFunctions>

#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
 #include <g2o/types/slam3d/se3quat.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

void prepareDataTUM(const std::string& dataFolder)
{
    // GT pose
    std::ifstream fin(dataFolder + "/poses.txt");
    std::ofstream fo(dataFolder + "/../data_tum/pose_gt.txt");
    if(!fin.is_open() || !fo.is_open()) return;
    int i = 1;
    while(!fin.eof())
    {
        Eigen::Matrix<double, 3, 4> pose;
        fin >> pose(0,0) >> pose(0,1) >> pose(0,2) >> pose(0,3) >> pose(1,0) >> pose(1,1) >> pose(1,2) >> pose(1,3) >> pose(2,0) >> pose(2,1) >> pose(2,2) >> pose(2,3);
        if(fin.fail()) break;
        Eigen::Matrix3d R = pose.block(0,0,3,3);
        Eigen::Quaterniond q(R);
        fo << i << " " << pose(0,3) << " " << pose(1,3) << " " << pose(2,3) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
        i++;

    }
    fin.close();
    fo.close();

    // estimate pose，landmark
    fin.open(dataFolder + "/hidden_state.txt");
    fo.open(dataFolder + "/../data_tum/pose_estimate.txt");
    std::ofstream fo_2(dataFolder + "/../data_tum/landmark_estimate.txt");
    if(!fin.is_open() || !fo.is_open() || !fo_2.is_open()) return;
    i = 1;
    int j = 1;

    // // -0.234382 -0.141915 4.29134 0.00289189 -0.00516178 -0.00130804 0.999982
    // Eigen::Quaterniond q = Eigen::Quaterniond(0.999982, 0.00289189, -0.00516178, -0.00130804);
    // Eigen::Isometry3d gt_0;
    // gt_0.linear() = q.matrix();
    // gt_0.translation() = Eigen::Vector3d(-0.234382, -0.141915, 4.29134);
    // // -0.0197301 -0.00523678 1.20073 0.00343222 -0.00981528 -0.00533388 0.999932
    // q = Eigen::Quaterniond(0.999932, 0.00343222, -0.00981528, -0.00533388);
    // Eigen::Isometry3d esti_0;
    // esti_0.linear() = q.matrix();
    // esti_0.translation() = Eigen::Vector3d(-0.0197301, -0.00523678, 1.20073);
    // std::cout << gt_0.matrix() << std::endl;
    // std::cout << esti_0.matrix() << std::endl;
    // std::cout << esti_0.matrix().inverse() << std::endl;
    // Eigen::Matrix4d T_gt_esti = gt_0.matrix() * esti_0.matrix().inverse();
    // std::cout << T_gt_esti << std::endl;

    while(!fin.eof())
    {
        if(i <= 250)
        {
            Eigen::Vector3d v, w;
            fin >> v(0) >> v(1) >> v(2) >> w(0) >> w(1) >> w(2);
            // std::cout << v(0) << " " << v(1) << std::endl;
            Eigen::Matrix4d se;
            se << 0, -w(2), w(1), v(0),
                w(2), 0, -w(0), v(1),
                -w(1), w(0), 0, v(2),
                0,0,0,0;
            Eigen::Matrix4d H = se.exp();
            // std::cout << H << std::endl;
            Eigen::Matrix3d R = H.block(0,0,3,3);
            Eigen::Quaterniond q(R);
            fo << i << " " << H(0,3) << " " << H(1,3) << " " << H(2,3) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
        }
        else
        {
            double x, y, z;
            fin >> x >> y >> z;
            fo_2 << j << " " << x << " " << y << " " << z << " " << std::endl;
            j++;
        }
        i++;
        // break;
    }
    fin.close();
    fo.close();
    fo_2.close();

    // observations，2d、3d index
    fin.open(dataFolder + "/observations.txt");
    fo.open(dataFolder + "/../data_tum/observations.txt");
    if(!fin.is_open() || !fo.is_open()) return;
    int num_frames, num_landmarks;
    fin >> num_frames >> num_landmarks;
    fo << num_frames << std::endl;
    fo << num_landmarks << std::endl;
    while(!fin.eof())
    {
        int num_kps;
        float r, c;
        int landmark_index;
        fin >> num_kps;
        std::cout << num_kps << std::endl;
        if(fin.fail()) break;
        fo << num_kps << std::endl;
        std::vector<float> rs, cs;
        for(int i = 0; i < num_kps; i++)
        {
            fin >> r >> c;
            rs.push_back(r);
            cs.push_back(c);
        }
        for(int i = 0; i < num_kps; i++)
        {
            fin >> landmark_index;
            fo << rs[i] << " " << cs[i] << " " << landmark_index << std::endl;
        }
    }
    fin.close();
    fo.close();

}

void loadData(std::vector<Eigen::Isometry3d>& gtPose, 
              std::vector<Eigen::Isometry3d>& estiPose,
              std::vector<Eigen::Vector3d>& estiLandmark,
              std::vector<Eigen::Matrix<double, -1, 2>>& observationKps,
              std::vector<Eigen::Matrix<int, -1, 1>>& observationLandmarkInds)
{
    gtPose.clear();
    estiPose.clear();
    estiLandmark.clear();
    observationKps.clear();
    observationLandmarkInds.clear();

    std::ifstream fin("../data_tum/pose_gt.txt");
    if(!fin.is_open()) return;
    while(!fin.eof())
    {
        int id;
        double tx, ty, tz, qx, qy, qz, qw;
        fin >> id >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
        if(fin.fail()) break;
        Eigen::Quaterniond q = Eigen::Quaterniond(qw, qx, qy, qz);
        Eigen::Isometry3d gt;
        gt.linear() = q.matrix();
        gt.translation() = Eigen::Vector3d(tx, ty, tz);
        gtPose.push_back(gt);
    }
    fin.close();
    std::cout << gtPose.size() << std::endl;

    fin.open("../data_tum/pose_estimate.txt");
    if(!fin.is_open()) return;
    while(!fin.eof())
    {
        int id;
        double tx, ty, tz, qx, qy, qz, qw;
        fin >> id >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
        if(fin.fail()) break;
        Eigen::Quaterniond q = Eigen::Quaterniond(qw, qx, qy, qz);
        Eigen::Isometry3d pose;
        pose.linear() = q.matrix();
        pose.translation() = Eigen::Vector3d(tx, ty, tz);
        estiPose.push_back(pose);
    }
    fin.close();
    std::cout << estiPose.size() << std::endl;

    fin.open("../data_tum/landmark_estimate.txt");
    if(!fin.is_open()) return;
    while(!fin.eof())
    {
        int id;
        double x, y, z;
        fin >> id >> x >> y >> z;
        if(fin.fail()) break;
        estiLandmark.push_back(Eigen::Vector3d(x, y, z));
    }
    fin.close();
    std::cout << estiLandmark.size() << std::endl;

    fin.open("../data_tum/observations.txt");
    if(!fin.is_open()) return;
    int num_frames, num_landmarks;
    fin >> num_frames >> num_landmarks;
    while(!fin.eof())
    {
        int num_kps;
        float r, c;
        int landmark_index;
        fin >> num_kps;
        if(fin.fail()) break;
        Eigen::Matrix<double, -1, 2> kps;
        Eigen::Matrix<int, -1, 1> landmarkInds;
        kps.resize(num_kps, 2);
        landmarkInds.resize(num_kps, 1);
        for(int i = 0; i < num_kps; i++)
        {
            fin >> r >> c >> landmark_index;
            kps(i, 0) = r;
            kps(i, 1) = c;
            landmarkInds(i) = landmark_index;
        }
        observationKps.push_back(kps);
        observationLandmarkInds.push_back(landmarkInds);
    }
    fin.close();
}

double cx = 6.071928000000e+02;
double cy = 1.852157000000e+02;
double fx = 7.188560000000e+02;
double fy = 7.188560000000e+02;

int main(int argc, char** argv)
{
    // prepareDataTUM("../data");

    std::vector<Eigen::Isometry3d> gtPose;
    std::vector<Eigen::Isometry3d> estiPose;
    std::vector<Eigen::Isometry3d> optPose;
    std::vector<Eigen::Vector3d> estiLandmark;
    std::vector<Eigen::Matrix<double, -1, 2> > observationKps;
    std::vector<Eigen::Matrix<int, -1, 1> > observationLandmarkInds;
    loadData(gtPose, estiPose, estiLandmark, observationKps, observationLandmarkInds);

    // 优化
    g2o::BlockSolver_6_3::LinearSolverType* linearSolver = new g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType>();
    g2o::BlockSolver_6_3* blockSolver = new g2o::BlockSolver_6_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* algo = new g2o::OptimizationAlgorithmLevenberg(blockSolver);

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(algo);
    optimizer.setVerbose(false);

    int num_poses = gtPose.size();
    int num_landmarks = estiLandmark.size();
    // pose顶点
    for(int i = 0; i < num_poses; i++)
    {
        g2o::VertexSE3Expmap* v = new g2o::VertexSE3Expmap();
        v->setId(i);
        if(i == 0)
            v->setFixed(true);
        v->setEstimate(g2o::SE3Quat(estiPose[i].linear(), estiPose[i].translation()));
        optimizer.addVertex(v);
    }

    // landmark顶点
    for(int i = 0; i < num_landmarks; i++)
    {
        g2o::VertexSBAPointXYZ* v = new g2o::VertexSBAPointXYZ();
        v->setId(num_poses + i);
        // v->setMarginalized(true);
        v->setEstimate(estiLandmark[i]);
        optimizer.addVertex(v);
    }

    g2o::CameraParameters* camera = new g2o::CameraParameters(fx, Eigen::Vector2d(cx, cy), 0);
    camera->setId(0);
    optimizer.addParameter(camera);

    // 边
    for(int i = 0; i < observationKps.size(); i++)
    {
        for(int j = 0; j < observationKps[i].rows(); j++)
        {
            double x = observationKps[i](j, 1);
            double y = observationKps[i](j, 0);
            int landmark_ind = observationLandmarkInds[i](j) - 1;
            g2o::EdgeProjectXYZ2UV* e = new g2o::EdgeProjectXYZ2UV();
            e->setVertex(0, dynamic_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(num_poses + landmark_ind)));
            e->setVertex(1, dynamic_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(i)));
            e->setMeasurement(Eigen::Vector2d(x, y));
            e->setInformation(Eigen::Matrix2d::Identity());
            e->setParameterId(0, 0);
            e->setRobustKernel(new g2o::RobustKernelHuber());
            optimizer.addEdge(e);
        }
    }

    std::cout << "vertex:" << optimizer.vertices().size() << ", edge:" << optimizer.edges().size() << std::endl;
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(20);

    std::ofstream fo("../data_tum/pose_opt.txt");
    if(!fo.is_open()) return -1;
    // 优化后pose
    for(int i = 0; i < num_poses; i++)
    {
        g2o::VertexSE3Expmap* v = dynamic_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(i));
        Eigen::Isometry3d pose = v->estimate();
        double tx = pose.translation()(0);
        double ty = pose.translation()(1);
        double tz = pose.translation()(2);
        Eigen::Quaterniond q(pose.rotation());
        double qx = q.x();
        double qy = q.y();
        double qz = q.z();
        double qw = q.w();
        fo << i + 1 << " " << tx << " "  << ty << " "  << tz << " "  << qx << " "  << qy << " "  << qz << " "  << qw << std::endl;
    }
    fo.close();

    // 优化后landmark
    for(int i = 0; i < num_landmarks; i++)
    {

    }

    return 0;
}