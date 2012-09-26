/* 
* OpenCL Raytracer
* Copyright (C) 2011-2012 Cyrille Favreau <cyrille_favreau@hotmail.com>
*
* This library is free software; you can redistribute it and/or
* modify it under the terms of the GNU Library General Public
* License as published by the Free Software Foundation; either
* version 2 of the License, or (at your option) any later version.
*
* This library is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
* Library General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program.  If not, see <http://www.gnu.org/licenses/>. 
*/

/*
* Author: Cyrille Favreau <cyrille_favreau@hotmail.com>
*
*/
#include <fstream>
#include <vector>
#include <iostream>
#include <stdlib.h>

#include "PDBReader.h"

struct Atom
{
   float4 position;
   int    boxId;
   int    materialId;
};

PDBReader::PDBReader(void) : m_nbBoxes(0), m_nbPrimitives(0)
{
}

PDBReader::~PDBReader(void)
{
}

void PDBReader::loadAtomsFromFile(
   const std::string& filename,
   CudaKernel& cudaKernel,
   int boxId )
{
   std::vector<Atom> atoms;
   float4 minPos = {  100000.f,  100000.f,  100000.f, 0.f };
   float4 maxPos = { -100000.f, -100000.f, -100000.f, 0.f };
   int maxBoxes = 0;

   std::ifstream file(filename.c_str());
   if( file.is_open() )
   {
      while( file.good() )
      {
         std::string line;
         std::getline( file, line );
         if( line.find("ATOM") == 0 )
         {
            Atom atom;
            std::string atomName;
            std::string value;
            int i(0);
            while( i<line.length() )
            {
               switch(i)
               {
               case 12: // Atom name
               case 22: // Box Id
               case 30: // x
               case 38: // y
               case 46: // z
                  value = "";
                  break;
               case 16: atomName = value; break;
               case 26: 
                  atom.boxId = static_cast<int>(atoi(value.c_str())); 
                  maxBoxes = (atom.boxId>maxBoxes) ? atom.boxId : maxBoxes;
                  break;
               case 37: atom.position.x = static_cast<float>(atof(value.c_str())); break;
               case 45: atom.position.y = static_cast<float>(atof(value.c_str())); break;
               case 53: atom.position.z = static_cast<float>(atof(value.c_str())); break;
               default:
                  if( line.at(i) != ' ' ) value += line.at(i);
               }
               i++;
            }

            // Material
            atom.materialId = 2;
            atom.materialId += ( atomName == "N"   ) ?  1 : 0;
            atom.materialId += ( atomName == "CA"  ) ?  2 : 0;
            atom.materialId += ( atomName == "C"   ) ?  3 : 0;
            atom.materialId += ( atomName == "O"   ) ?  4 : 0;
            atom.materialId += ( atomName == "CB"  ) ?  5 : 0;
            atom.materialId += ( atomName == "CG"  ) ?  6 : 0;
            atom.materialId += ( atomName == "OD1" ) ?  7 : 0;
            atom.materialId += ( atomName == "OD2" ) ?  8 : 0;
            atom.materialId += ( atomName == "SD"  ) ?  9 : 0;
            atom.materialId += ( atomName == "CE"  ) ? 10 : 0;
            atom.materialId += ( atomName == "OE1" ) ? 11 : 0;

            // Radius
            atom.position.w = 8.f; //center.w; TODO: Radius
            atom.position.w *= ( atomName == "N"   ) ?  2.f : 1.f;
            atom.position.w *= ( atomName == "CA"  ) ?  3.f : 1.f;
            atom.position.w *= ( atomName == "C"   ) ?  4.f : 1.f;
            atom.position.w *= ( atomName == "O"   ) ?  5.f : 1.f;
            atom.position.w *= ( atomName == "CB"  ) ?  6.f : 1.f;
            atom.position.w *= ( atomName == "CG"  ) ?  7.f : 1.f;
            atom.position.w *= ( atomName == "OD1" ) ?  8.f : 1.f;
            atom.position.w *= ( atomName == "OD2" ) ?  9.f : 1.f;
            atom.position.w *= ( atomName == "SD"  ) ? 10.f : 1.f;
            atom.position.w *= ( atomName == "CE"  ) ? 11.f : 1.f;
            atom.position.w *= ( atomName == "OE1" ) ? 12.f : 1.f;

            // min
            minPos.x = (atom.position.x < minPos.x) ? atom.position.x : minPos.x;
            minPos.y = (atom.position.y < minPos.y) ? atom.position.y : minPos.y;
            minPos.z = (atom.position.z < minPos.z) ? atom.position.z : minPos.z;

            // max
            maxPos.x = (atom.position.x > maxPos.x) ? atom.position.x : maxPos.x;
            maxPos.y = (atom.position.y > maxPos.y) ? atom.position.y : maxPos.y;
            maxPos.z = (atom.position.z > maxPos.z) ? atom.position.z : maxPos.z;
            
            // add Atom to the list
            atoms.push_back(atom);
         }
      }
      file.close();
   }

   float4 center;
   center.x = (minPos.x+maxPos.x)/2.f;
   center.y = (minPos.y+maxPos.y)/2.f;
   center.z = (minPos.z+maxPos.z)/2.f;

   int currentBox(boxId);
   int counter(0);
   int nbPrimitivesPerBox = static_cast<int>((atoms.size()+NB_MAX_BOXES)/(NB_MAX_BOXES-boxId));

   int n = 25;
   nbPrimitivesPerBox = (nbPrimitivesPerBox < n) ? n : nbPrimitivesPerBox;
   m_nbPrimitives = 0;

   std::vector<Atom>::const_iterator it = atoms.begin();
   while( it != atoms.end() )
   {
      Atom atom(*it);
      int nb = cudaKernel.addPrimitive( ptSphere );
      cudaKernel.setPrimitive( 
         nb, 
         currentBox, //atom.boxId,
         50.f*(atom.position.x - center.x), 
         50.f*(atom.position.y - center.y), 
         50.f*(atom.position.z - center.z), 
         atom.position.w, 0.f, 
         atom.materialId, 1, 1 );
      ++it;
      ++counter;

      if( counter%nbPrimitivesPerBox==0 ) currentBox++;
      //if( counter%NB_MAX_PRIMITIVES_PER_BOX==0 ) currentBox++;
   }
   m_nbPrimitives = static_cast<int>(atoms.size());

   m_nbBoxes = currentBox;
   std::cout << "-==========================================================-" << std::endl;
   std::cout << "filename: " << filename << std::endl;
   std::cout << "------------------------------------------------------------" << std::endl;
   std::cout << "Number of atoms    : " << m_nbPrimitives << std::endl;
   std::cout << "Number of boxes    : " << m_nbBoxes << std::endl;
   std::cout << "Number of atoms/box: " << nbPrimitivesPerBox << std::endl;
   std::cout << "------------------------------------------------------------" << std::endl;
}
