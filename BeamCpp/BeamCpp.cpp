// BeamCpp.cpp : Diese Datei enthält die Funktion "main". Hier beginnt und endet die Ausführung des Programms.
//

#include <iostream>
#include <algorithm>
#include <ranges>
#include <Eigen/Dense>

#include "opennn.h"     
#include "loss_index.h"

using namespace opennn;

constexpr auto L = double{ 1.0 };
constexpr auto E = double{ 210e9 };
constexpr auto I = double{ 1e-6 };
constexpr auto q = double{ 1e3 };

constexpr auto n_elem = size_t{ 10 }; // number of element
constexpr auto n_nodes = size_t{ n_elem + 1 }; // number of nodes

constexpr auto l_e = double{ L / static_cast<double>(n_elem) };

using namespace Eigen;

static MatrixX4d get_stiffness_matrix(const double L, const double E, const double I) {
    Matrix4d A;
    A << 12, 6 * L, -12, 6 * L,
        6 * L, 4 * L * L, -6 * L, 2 * L * L,
        -12, -6 * L, 12, -6 * L,
        6 * L, 2 * L * L, -6 * L, 4 * L * L;
    A *= E * I / L / L / L;

    return A;
}

static Vector4d get_load(const double L, const double q) {
    Eigen::Vector4d v;
    v << 1, L / 6, 1, -L / 6;
    v *= q * L / 2;
    return v;
}

int main()
{
    Matrix4d K0{ get_stiffness_matrix(L, E, I) };
    Vector4d f0{ get_load(L,q) };
    std::cout << K0 << std::endl;
    std::cout << f0 << std::endl;

    MatrixXd K{ MatrixXd::Zero(2 * n_nodes, 2 * n_nodes) };
    VectorXd f{ VectorXd::Zero(2 * n_nodes) }; 
    std::cout << K << std::endl;
    std::cout << f << std::endl;

    for (auto e : std::ranges::iota_view(0, static_cast<int>(n_elem))) {
        auto ke{ get_stiffness_matrix(l_e, E, I) };
        auto fe{ get_load(l_e,q) };

        // DOFs für dieses Element: [2*e, 2*e+1, 2*e+2, 2*e+3]
        std::array<int, 4> dofs = { 2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3 };

        // Lokale Steifigkeitsmatrix ke in globale K einsortieren
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                K(dofs[i], dofs[j]) += ke(i, j);
            }
            f(dofs[i]) += fe(i);
        }
    }
    
    int n = K.rows();

    MatrixXd K_ff{ K.block(2, 2, n - 2,n - 2) };
    VectorXd f_f{ f.tail(f.size() - 2) };

    VectorXd x = K_ff.colPivHouseholderQr().solve(f_f);

    std::cout << "K =\n" << K << "\n\n";
    std::cout << "f =\n" << f.transpose() << "\n";

    std::cout << "K_ff =\n" << K_ff << "\n\n";
    std::cout << "f_f =\n" << f_f.transpose() << "\n";

    std::cout << "\nx=\n";
    for (int j = 0; j < x.size(); ++j) {
        std::cout << x[j] << "\t";
        ++j;
    }

    auto exact_displacement = [=](double x) {
        return q / (24.0 * E * I) * x * x * (x * x - 4.0 * L * x + 6.0 * L * L);
        };

    double mse{};
    std::cout << "\nx=\n";
    for (double t{}; t < L;) {
        double y{ exact_displacement(t) };
        std::cout << y << "\t";
        t += l_e;
    }

    std::cout << "\ntheta=\n";
    for (int j = 1; j < x.size(); ++j) {
        std::cout << x[j] << "\t";
        ++j;
    }

    std::cout << "\nHello World!\n";
    std::cout << get_stiffness_matrix(L, E, I) << std::endl;
}

// Programm ausführen: STRG+F5 oder Menüeintrag "Debuggen" > "Starten ohne Debuggen starten"
// Programm debuggen: F5 oder "Debuggen" > Menü "Debuggen starten"

// Tipps für den Einstieg: 
//   1. Verwenden Sie das Projektmappen-Explorer-Fenster zum Hinzufügen/Verwalten von Dateien.
//   2. Verwenden Sie das Team Explorer-Fenster zum Herstellen einer Verbindung mit der Quellcodeverwaltung.
//   3. Verwenden Sie das Ausgabefenster, um die Buildausgabe und andere Nachrichten anzuzeigen.
//   4. Verwenden Sie das Fenster "Fehlerliste", um Fehler anzuzeigen.
//   5. Wechseln Sie zu "Projekt" > "Neues Element hinzufügen", um neue Codedateien zu erstellen, bzw. zu "Projekt" > "Vorhandenes Element hinzufügen", um dem Projekt vorhandene Codedateien hinzuzufügen.
//   6. Um dieses Projekt später erneut zu öffnen, wechseln Sie zu "Datei" > "Öffnen" > "Projekt", und wählen Sie die SLN-Datei aus.
