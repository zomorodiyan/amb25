/*--------------------------------*- C++ -*----------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Version:  10
     \\/     M anipulation  |
\*---------------------------------------------------------------------------*/
FoamFile
{
    format      ascii;
    class       dictionary;
    location    "constant";
    object      physicalProperties.water;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

viscosityModel  constant;


nu          4.3e-07; // IN718 kinematic viscosity [m^2/s]
rho         8190.0;  // IN718 density [Kg/m^3]

Tsolidus    1533.15;  // IN718 Solidus [K] (AMB2025 Ch.6&7 description file)
Tliquidus   1609.15;  // IN718 Liquidus[K] (AMB2025 Ch.6&7 description file)

LatentHeat  2.9e5;   // IN718 latent heat of fusion [J/kg]
beta        1.3e-5;  // IN718 thermal expansion coefficient [1/K]

poly_kappa  (25.0 0.0 0 0 0 0 0 0);  // IN718 thermal conductivity [W/m·K]
poly_cp     (435.0 0.0 0 0 0 0 0 0); // IN718 specific heat capacity [J/kg·K]

elec_resistivity	1e-6;
    
//Tsolidus 1658;
//Tliquidus 1723;


// ************************************************************************* //
