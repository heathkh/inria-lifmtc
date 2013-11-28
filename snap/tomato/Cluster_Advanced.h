//----------------------------------------------------------------------
//----------------------------------------------------------------------
// File:		Cluster_Basic.h
// Programmer:		Primoz Skraba
// Description:		Basic Cluster data structure
// Last modified:	Sept 8, 2009 (Version 0.1)
//----------------------------------------------------------------------
//  Copyright (c) 2009 Primoz Skraba.  All Rights Reserved.
//-----------------------------------------------------------------------
//
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
//
//-----------------------------------------------------------------------
//----------------------------------------------------------------------
// History:
//	Revision 0.1  August 10, 2009
//		Initial release
//----------------------------------------------------------------------
//----------------------------------------------------------------------

#ifndef __CLUSTER__BASIC__H
#define __CLUSTER__BASIC__H

#include <iostream>
#include <fstream>
#include <cmath>
#include <ctime>
#include <algorithm>

#include "Cluster.h"

class PersistanceDiagramPoint {
public:
  PersistanceDiagramPoint(){}
  PersistanceDiagramPoint(double birth, double death) :
    birth(birth),
    death(death) {
  }
  double birth;
  double death;
};

class Cluster {
public:
  Cluster(){}
  Cluster(double birth, double death, double persistence) :
    birth(birth),
    death(death),
    persistence(persistence){
  }
	double birth;
	double death;
	double persistence;
	std::vector<int> member_indices;
};

bool OrderDecreasing (const Cluster& a, const Cluster& b) { return (b.persistence < a.persistence); }


//==============================
//basic implementation of 
//cluster class
//==============================
template<class Iterator>
class Clustering: public Cluster_Base<Iterator> {
private:
	std::vector<Interval> Int_Data;
	//--------------------------------
	// since we are storing it as a
	// vector, it is equivalent to
	// storing an iterator
	//--------------------------------
	std::map<Iterator, int> Generator;
	//--------------------------------
	//for faster searching
	//--------------------------------

public:
	double tau;

	//----------------------
	//create a new interval
	//----------------------
	void new_cluster(Iterator x) {
		Generator[x] = Int_Data.size();
		Int_Data.push_back(Interval(x->func()));

	}

	//----------------------
	// Merge two intervals
	// note this will only output
	// correctly if persistence
	// threshold is set to infty
	//----------------------
	bool merge(Iterator x, Iterator y) {
		// note: by hypothesis, y->func() >= x->func()
		assert(y->func() >= x->func());

		//---------------------------------
		// test prominences of both clusters
		// assumptions:
		//   - y is its cluster's root
		//   - x is attached to its root directly
		//---------------------------------
		if (std::min(x->get_sink()->func(), y->func()) < x->func() + tau) {
			//---------------------------------
			//kill younger interval
			//---------------------------------
			int i = Generator[x->get_sink()];
			int j = Generator[y];
			if (y->func() <= x->get_sink()->func()) {
				assert(Int_Data[j].inf());
				Int_Data[j].close(x->func());
			} else {
				assert(Int_Data[i].inf());
				Int_Data[i].close(x->func());
			}
			return true;
		}

		return false;
	}

	//----------------------------
	//unwieghted gradient choice
	//----------------------------
	Iterator gradient(Iterator x, std::set<Iterator> &List) {
		typename std::set<Iterator>::iterator it;
		Iterator y = x;
		//--------------------
		//find oldest neighbor
		//--------------------
		for (it = List.begin(); it != List.end(); it++) {
			if (*it < y)
				y = *it;
		}
		assert(y != x);
		return y;
	}

	//------------------------------------------------------------
	// weighted gradient choice --
	// need access to Distance struct
	// this simplest way to do this is to
	// derive from Cluster_Basic and   add a pointer to a
	// distance structure
	//------------------------------------------------------------
	// Iterator gradient(Iterator x,std::set<Iterator> &List){
	// double dist1 =...;
	// double dist2 =...;
	// return ((x->func()-y1->func())/dist1<(x->func()-y2->func())/dist2) ? y1 : y2;
	//}

	//-----------------------------------
	// This is a user-defined test
	// if you want to skip over
	// some nodes
	//-----------------------------------
	inline bool test(Iterator) {
		return true;
	}

	//-----------------------------------
	// Same thing but for valid
	// neighbors
	//-----------------------------------
	bool neighbor_test(Iterator x) {
		return !x->data.boundary_flag;
	}

	//-----------------------------------
	// check to make sure this is a true
	// peak
	//-----------------------------------
	bool valid_neighborhood(std::set<Iterator> &in) {
		typename std::set<Iterator>::iterator it;
		if (in.size() == 0)
			return true;

		for (it = in.begin(); it != in.end(); it++) {
			if (!((*it)->data.boundary_flag))
				return true;
		}
		return false;
	}


  //------------------------------
  // Output intervals
  //------------------------------
  void get_diagram(std::vector<PersistanceDiagramPoint>* diagram){
    std::vector<Interval>::iterator it;
    assert(diagram);
    diagram->clear();
    for(it = Int_Data.begin();it != Int_Data.end(); ++it){
      diagram->push_back(PersistanceDiagramPoint(it->birth(), it->inf() ? 0 : it->death()) );
    }
  }

	//------------------------------
	//
	//------------------------------
	template<class IIterator>
	void get_clusters(IIterator start, IIterator finish, std::vector<Cluster>* clusters) {
		//------------------------------
		// run through and
		// create map of prominent clusters
		//------------------------------
		std::map<Iterator, int> cluster_ids;
		assert(clusters);

		clusters->clear();

		int member_index = 0;
		for (IIterator it = start; it != finish; it++, member_index++) {
		  Iterator sink = find_sink(*it);

		  if (sink->func() >= tau){
		    typename std::map<Iterator, int>::iterator iter = cluster_ids.find(sink);

		    // create cluster if needed
		    int cluster_index = -1;
		    if (iter == cluster_ids.end()){
		      cluster_index = clusters->size();
		      cluster_ids[sink] = cluster_index;
		      const Interval& interval = Int_Data[Generator[sink]];

          double birth = interval.birth();
          double death = interval.inf() ? 0 : interval.death();
          double persistence =  birth - death;
          //std::cout << "persistence: " << persistence << std::endl;
          clusters->push_back(Cluster(birth, death, persistence));

		    }
		    else{
		      cluster_index = iter->second;
		    }
		    assert(cluster_index != -1);
		    clusters->at(cluster_index).member_indices.push_back(member_index);
		  }
		}

		std::sort(clusters->begin(), clusters->end(), OrderDecreasing);
	}


};

#endif
