
/* Author: Daekeun You (January, 2013) 
 * 
 * utilFunctions
 * 			a collection of functions for basic computation.  												
 * 	
 */

package gov.nih.nlm.iti.panelSegmentation;

//import com.googlecode.javacv.cpp.opencv_core.*;
//import com.googlecode.javacv.cpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.*;
import org.bytedeco.javacpp.opencv_core;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.lang.Math;
import java.util.ArrayList;

public class UtilFunctions{

	public UtilFunctions(){}
	
	public boolean decisionMade = false;
			
	static public class BBoxCoor{
		public int left, top, right, bottom;		
	}
			
	public int sorting_max_2_min(double [] a, double [] b, int len){

		int i, j, k;
		double max;
		int max_idx;

		k = 0;
		for(i=0; i<len; i++){
			max = -10000.0;
			max_idx = 0;
			for(j=0; j<len; j++){
				if( a[j] > max ){
					max = a[j];
					max_idx = j;
				}
			}
			a[max_idx] = -10000.0;
			b[k++] = max;
		}

		return 1;
	}


	public int sorting_max_2_min_w_index(double [] a, double [] b, int [] index, int len){

		int i, j, k;
		double max;
		int max_idx;

		k = 0;
		for(i=0; i<len; i++){
			max = -10000.0;
			max_idx = 0;
			for(j=0; j<len; j++){
				if( a[j] > max ){
					max = a[j];
					max_idx = j;
				}
			}
			a[max_idx] = -10000.0;
			index[k] = max_idx;
			b[k++] = max;
		
		}

		return 1;
	}


	public int sorting_max_2_min_no_sorted_data(double [] a, int [] index, int len){

		int i, j, k;
		double max;
		int max_idx;

		k = 0;
		for(i=0; i<len; i++){
			max = -10000.0;
			max_idx = 0;
			for(j=0; j<len; j++){
				if( a[j] > max ){
					max = a[j];
					max_idx = j;
				}
			}
			a[max_idx] = -10000.0;
			index[k++] = max_idx;	
		}

		return 1;
	}

	public int sorting_max_2_min_INT(int [] a, int [] index, int len){

		int i, j, k;
		int max;
		int max_idx;

		k = 0;
		for(i=0; i<len; i++){
			max = -10000;
			max_idx = 0;
			for(j=0; j<len; j++){
				if( a[j] > max ){
					max = a[j];
					max_idx = j;
				}
			}
			a[max_idx] = -10000;
			index[k++] = max_idx;	
		}

		return 1;
	}


	public int sorting_min_2_max(int [] a, int [] index, int len){

		int i, j, k;
		int min;
		int min_idx;

		k = 0;
		for(i=0; i<len; i++){
			min = 10000;
			min_idx = 0;
			for(j=0; j<len; j++){
				if( a[j] < min ){
					min = a[j];
					min_idx = j;
				}
			}
			a[min_idx] = 10000;
			index[k++] = min_idx;	
		}

		return 1;
	}


	public int negative_image(IplImage img){
		int i, j;
		int height, width;
		CvMat srcMat = img.asCvMat();
		
		height = img.height();
		width = img.width();
		
	//	CvMat negativeMat = cvCreateMat(srcMat.rows(), srcMat.cols(), CV_8UC1);
	//	cvSetZero(negativeMat);
				
		for(i=0; i<height; i++){
			for(j=0; j<width; j++){
				
				if( srcMat.get(i, j) > 0 )
					srcMat.put(i, j, 0);
				else
					srcMat.put(i, j, 255);
			}
		}

		img = srcMat.asIplImage();
		
		return 1;
	}
	

	public int cvImgThreshold(IplImage src, IplImage dst, int level){
		int i, j;
		int height, width;

		width = src.width();
		height = src.height();

		CvMat in_data = src.asCvMat();
		CvMat out_data = dst.asCvMat();

		for(i = 0; i<height; i++){
			for(j = 0; j<width; j++){
				if( in_data.get(i, j) > level )
					out_data.put(i, j, 255);
				else
					out_data.put(i, j, 0);
			}
		}

		return 1;
	}
	
	
	public int featureExtractionFromContour36(CvSeq contour, double [] feature, int featureDim, int featureType, PanelLabelDetection.objRect chBndBox){

		int left, top, right, bottom;
		int i;
		int mesh_width, mesh_height;
		int dir, dir_x, dir_y;
		CvSeq inner_contour = new CvSeq(null);
		double aspect_ratio;
		boolean is_narrow_char = false;
		
		int [] mesh_sum = new int [9];
		double [] comp_max = new double [4];
		
		left = 10000;
		right = 0;
		top = 10000;
		bottom = 0;
		CvPoint pt, pt_prev, pt_next;

		this.decisionMade = false;
		
		for(i=0; i<contour.total(); i++){
			pt = new CvPoint(opencv_core.cvGetSeqElem(contour, i));

			if( pt.x() > right )
				right = pt.x();
			if( pt.x() < left )
				left = pt.x();
			if( pt.y() > bottom )
				bottom = pt.y();
			if( pt.y() < top )
				top = pt.y();
		}
		
		chBndBox.left = left;
			
		chBndBox.bottom = bottom;
		chBndBox.right = right;
		chBndBox.top = top;

		mesh_width = (right - left + 1)/3;
		mesh_height = (bottom - top + 1)/3;
		
		if( mesh_width < 3 || mesh_height < 3 )
			return -1;
		
		aspect_ratio = (right-left)/(double)(bottom-top);
		
		// this can't be letter.
		if( aspect_ratio >= 1.5 )
			return -1;

//		if( aspect_ratio < 0.3 )
//			is_narrow_char = 1;

		// process exterior contour first.
		pt_prev = new CvPoint(opencv_core.cvGetSeqElem(contour, 0));
		
		for(i=1; i<contour.total(); i++){
			pt_next = new CvPoint(opencv_core.cvGetSeqElem(contour, i));

			dir = -1;
			if( pt_prev.x() == pt_next.x() )
				dir = 0;

			if( pt_prev.y() == pt_next.y() && dir == -1 )
				dir = 2;

			if( pt_prev.y() > pt_next.y() && dir == -1 ){
				if( pt_prev.x() > pt_next.x() )
					dir = 3;
				else
					dir = 1;
			}

			if( pt_prev.y() < pt_next.y() && dir == -1 ){
				if( pt_prev.x() > pt_next.x() )
					dir = 1;
				else
					dir = 3;
			}
			
			dir_x = (pt_prev.x() - left)/mesh_width;
			dir_y = (pt_prev.y() - top)/mesh_height;
			
			if( dir_x > 2 )
				dir_x = 2;
			if( dir_y > 2 )
				dir_y = 2;

			if( is_narrow_char )
				dir_x = 1;

			feature[12*dir_y + 4*dir_x + dir] += 1.0;

			pt_prev.put(pt_next);			
		
		}

		inner_contour = contour.v_next();
		
		while( inner_contour != null ){

			if( inner_contour.total() < 10 ){
				inner_contour = inner_contour.h_next();
				continue;
			}

			this.decisionMade  = true;
			pt_prev = new CvPoint(opencv_core.cvGetSeqElem(inner_contour, 0));
			
			for(i=1; i<inner_contour.total(); i++){
				pt_next = new CvPoint(opencv_core.cvGetSeqElem(inner_contour, i));
					
				dir = -1;
				if( pt_prev.x() == pt_next.x() )
					dir = 0;

				if( pt_prev.y() == pt_next.y() && dir == -1 )
					dir = 2;

				if( pt_prev.y() > pt_next.y() && dir == -1 ){
					if( pt_prev.x() > pt_next.x() )
						dir = 3;
					else
						dir = 1;
				}

				if( pt_prev.y() < pt_next.y() && dir == -1 ){
					if( pt_prev.x() > pt_next.x() )
						dir = 1;
					else
						dir = 3;
				}
				
				dir_x = (pt_prev.x() - left)/mesh_width;
				dir_y = (pt_prev.y() - top)/mesh_height;
				
				if( dir_x > 2 )
					dir_x = 2;
				if( dir_y > 2 )
					dir_y = 2;

				if( is_narrow_char )
					dir_x = 1;

				feature[12*dir_y + 4*dir_x + dir] += 1.0;

				pt_prev.put(pt_next);
				
			
			}

			inner_contour = inner_contour.h_next();
		}
				
		// Normalize

		for(i=0; i<36; i++)
			mesh_sum[i/4] += (int)feature[i];
				
		for(i=0; i<36; i++){
			if( mesh_sum[i/4] > 0 )
				feature[i] /= (mesh_sum[i/4]*1.0);					
		}

		for(i=0; i<36; i++){
			if( feature[i] > comp_max[i%4] )
				comp_max[i%4] = feature[i];
		}

		for(i=0; i<36; i++){
			if( comp_max[i%4] > 0 )
				feature[i] /= comp_max[i%4];
		}

		return 1;
	}


	public int drawContour(IplImage image, CvSeq contour, int intensity){
		int k;
		CvPoint pt;
		
		CvMat srcMat = image.asCvMat();

		for(k = 0; k < contour.total(); k++){
			pt = new CvPoint(opencv_core.cvGetSeqElem(contour, k));
			
			srcMat.put(pt.y(), pt.x(), intensity);			
		}
		
		image = srcMat.asIplImage();

		return 1;
	}
	
	
    public void evaluatePanelSplittingResult(ArrayList<PanelSplitter.Final_Panel> final_panel_info, String ground_fname, PanelSplitter.evalCount evalResult){
		String digitStr = "0123456789";
		String lowercase = "abcdefghijklmnopqrstuvwxyz";
		
    	String gt_line = "";
		int gt_panel_no;
		String label_str = "";
		int [] tmp_coor = new int[8];
		int min_sum, max_sum;
		int min_idx, max_idx;
		int matching_no, best_match_idx;
		double best_match_ratio;
		float cur_ratio;
	
		ArrayList<PanelSplitter.Final_Panel> ground_panel_info = new ArrayList<PanelSplitter.Final_Panel>();
				
		boolean lower_upper;
		
		File gt_file = new File(ground_fname);
		
		// no ground truth file available. 
		if( gt_file.exists() != true ){		
			System.out.println(ground_fname + "doesn't exist");
			return;
		}
		
		lower_upper = true;
	
		FileReader file = null;
		try {
			file = new FileReader(ground_fname);
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		BufferedReader reader = new BufferedReader(file);
		
		gt_line = "";		
		try {
			gt_line = reader.readLine();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		String tmp_gt_line = gt_line;
		
		String [] split_ = tmp_gt_line.split("<name>Label ");
		
		for(int k = 1; k< split_.length; k++){
			if( lowercase.indexOf(split_[k].charAt(0)) != -1 )
				lower_upper = false;			
		}		
		
		
		gt_panel_no = 0;
		
		// read ground truth info
		// panel coor and label.
								
		
		String [] splitStr = gt_line.split("<name>Panel ");
		
		if( splitStr.length > 1 ){
			for(int k=1; k<splitStr.length; k++){
				label_str = "";
				label_str = splitStr[k].split("</name>")[0];
				
				if( label_str.length() > 1 ){
					continue;
				}
				
				if( label_str.length() > 0 && digitStr.indexOf(label_str.charAt(0)) != -1 ){
					continue;
				}
				
				PanelSplitter.Final_Panel tmp_gt_info = new PanelSplitter.Final_Panel();
				tmp_gt_info.label = label_str.charAt(0);
				
				String [] xStr = splitStr[k].split("<x>");
				
				for(int i = 1; i<Math.min(xStr.length, 5); i++){
					tmp_coor[(i-1)*2] = Integer.parseInt(xStr[i].split("</x>")[0]);
				}
				
				String [] yStr = splitStr[k].split("<y>");
				
				for(int i = 1; i<Math.min(yStr.length, 5); i++){
					tmp_coor[(i-1)*2+1] = Integer.parseInt(yStr[i].split("</y>")[0]);
				}
				
				
				min_sum = 10000;
				max_sum = 0;
				min_idx = 0;
				max_idx = 2;
		
				for(int i=0; i<4; i++){
					if( tmp_coor[i*2]+tmp_coor[i*2+1] > max_sum ){
						max_sum = tmp_coor[i*2]+tmp_coor[i*2+1];
						max_idx = i;
					}
		
					if( tmp_coor[i*2]+tmp_coor[i*2+1] < min_sum ){
						min_sum = tmp_coor[i*2]+tmp_coor[i*2+1];
						min_idx = i;
					}
				}
				
				 
				tmp_gt_info.left = tmp_coor[min_idx*2];
				tmp_gt_info.top = tmp_coor[min_idx*2+1];
				tmp_gt_info.right = tmp_coor[max_idx*2];
				tmp_gt_info.bottom = tmp_coor[max_idx*2+1];
		
				ground_panel_info.add(tmp_gt_info);
			}					
		
		}
		
		evalResult.total_gt_panels += ground_panel_info.size();
		evalResult.total_detected_panels += final_panel_info.size();
		
		// convert lower<->upper cases if mismatchs
		
		if( ground_panel_info.size() > 0 ){
			if( lower_upper == true && ground_panel_info.get(0).label >= 97 ){
				for(int i=0; i<ground_panel_info.size(); i++){
					ground_panel_info.get(i).label -= 32;
				}
			}
		
			if( lower_upper == false && ground_panel_info.get(0).label < 97 ){
				for(int i=0; i<gt_panel_no; i++){
					ground_panel_info.get(i).label += 32;
				}
			}
		}
		
		
		matching_no = 0;
		for(int i=0; i<ground_panel_info.size(); i++){
			best_match_ratio = 0.0;
			best_match_idx = 0;
			for(int j=0; j<final_panel_info.size(); j++){
				if( final_panel_info.get(j).matched == true )
					continue;

				if( Math.min(final_panel_info.get(j).right, ground_panel_info.get(i).right) > Math.max(final_panel_info.get(j).left, ground_panel_info.get(i).left) &&
					Math.min(final_panel_info.get(j).bottom, ground_panel_info.get(i).bottom) > Math.max(final_panel_info.get(j).top, ground_panel_info.get(i).top) ){

					cur_ratio = (Math.min(final_panel_info.get(j).right, ground_panel_info.get(i).right)-Math.max(final_panel_info.get(j).left, ground_panel_info.get(i).left))*(Math.min(final_panel_info.get(j).bottom, ground_panel_info.get(i).bottom)-Math.max(final_panel_info.get(j).top, ground_panel_info.get(i).top));

					if( cur_ratio > best_match_ratio ){
						best_match_ratio = cur_ratio;
						best_match_idx = j;
					}
				}
			}

			// found a matching detected panel with current ground truth panel.
			if( best_match_ratio >= 0.75*(ground_panel_info.get(i).right - ground_panel_info.get(i).left)*(ground_panel_info.get(i).bottom - ground_panel_info.get(i).top) && final_panel_info.get(best_match_idx).label == ground_panel_info.get(i).label ){
				final_panel_info.get(best_match_idx).matched = true;
				matching_no++;
			}
		}

		evalResult.total_match_panels += matching_no;

    }
    
    
}