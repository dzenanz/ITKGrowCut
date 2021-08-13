/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2021 Scientific Computing and Imaging Institute,
   University of Utah.


   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
 */
// Adapted from: https://github.com/ljzhu/FastGrowCut

#ifndef FastGrowCut_h
#define FastGrowCut_h

#include <math.h>
#include <queue>
#include <set>
#include <vector>
#include <stdlib.h>
#include <fstream>
#include <iterator>
#include <limits>

#include "FibHeap.h"

#include "itkImage.h"

namespace FGC
{

const NodeKeyValueType DIST_INF = std::numeric_limits<float>::max();
const NodeKeyValueType DIST_EPSILON = 1e-3f;
const unsigned char    NNGBH = 26;
using DistancePixelType = float;


template <typename IntensityPixelType, typename LabelPixelType>
class FastGrowCut
{
public:
  ~FastGrowCut() { this->Reset(); }

  void
  Reset();

  bool
  InitializationAHP(double distancePenalty);

  void
  DijkstraBasedClassificationAHP();

  bool
  ExecuteGrowCut(double distancePenalty);

  using LabelImageType = itk::Image<LabelPixelType, 3>;
  using DistanceImageType = itk::Image<DistancePixelType, 3>;

  typename LabelImageType::Pointer    m_ResultLabelVolume = LabelImageType::New();
  typename DistanceImageType::Pointer m_DistanceVolume = DistanceImageType::New();

  NodeIndexType m_DimX;
  NodeIndexType m_DimY;
  NodeIndexType m_DimZ;

  std::vector<NodeIndexType> m_NeighborIndexOffsets;
  std::vector<double>        m_NeighborDistancePenalties;
  std::vector<unsigned char> m_NumberOfNeighbors; // same everywhere except at the image boundary

  FibHeap *     m_Heap{ nullptr };
  FibHeapNode * m_HeapNodes{ nullptr }; // a node is stored for each voxel
  bool          m_bSegInitialized{ false };
  double        m_DistancePenalty{ 0.0 };
};
} // end namespace FGC

#include "FastGrowCut.hxx"

#endif // FastGrowCut_h
