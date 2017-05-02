
/* Author: Daekeun You (January, 2013) 
 * 
 * panelSplitter
 *  												
 * 	Input: 
 *  		String imgDir: Directory name containing images
 *  		String outDir: Directory name used to save split panels
 *  		String fileName: image name and extension only, no directory name included
 *  		String OCR_model_fname: Full path of NN_OCR.model file.
 *  		int no_text_panel: number of text labels (from Emilia's result)
 *  		String text_labels: labels from Emilia's result, comma(,) separated labels (e.g., [A,B,C]) 
 *			ArrayList<Integer> rectangles: array of integer numbers of panel rectangles from Jaylene's result.
 * 
 * 		* imgDir and outDir may or may not contain '/' at the end of the string. It doesn't matter.  
 * 
 *  Return: 
 *  		String returnStr: contains # of split panels and labels. ':::' separated.
 *  						  e.g., 3:::A,C,B
 *  
 *  Necessary external libraries:
 *  		JavaCV - several jar files
 *  		OpenCV - installed or core library files
 *  		Weka - weka.jar 
 * 
 *  
 *  Update history:
 *  
 *  		Jan. 16, 2014: Fixed garbage 'z' or 'Z' label problem by modifying code for assigning labels to unlabeled panels.
 *  		Feb. 19, 2014: set upper limit of image rectangles to 40. if more than 40, then no split.
 *			May 2, 2017: Jie Zou
 *					use mvn, JavaCV 1.3 and weka 3.6.14
 */


package gov.nih.nlm.iti.panelSegmentation;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.bytedeco.javacpp.opencv_imgcodecs.*;

import java.util.ArrayList;
import java.util.Arrays;

import weka.classifiers.Classifier;

public class PanelSplitter{
	
	
	static public class Final_Panel{
		public int left, top, right, bottom;
		public char label;
		public boolean matched;
	}
	
	static public class evalCount{
		public int total_gt_panels, total_detected_panels, total_match_panels;		
	}
		
	public PanelSplitter(){}
	
	UtilFunctions utilFunc = new UtilFunctions();

	String alphaStr = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
	
	
	
	public String panelSplitter(String imgDir,
							 String outDir,
							 String fileName,
							 Classifier OCR_model,
							 int no_text_panel, String text_labels, 
							 ArrayList<Integer> rectangles/*,			// for debug only		  
							 Core.PanelSplitter.evalCount evalNumbers*/) {

		int i, j, k, idx;
		int height, width;
		int setNo, match_label_no, unmatch_label_no;
		
		int [] label_order = new int [26];
		int [] ori_label_idx = new int [26];
		int [] sorted_label_idx = new int [26];		// initialize with -1
		String final_label_list = "";

		ArrayList<Final_Panel> finalSplitPanels = new ArrayList<Final_Panel>();
		
		String imgName = "";
		
		if( imgDir.charAt(imgDir.length()-1) == '/' )
			imgName = imgDir + fileName;
		else
			imgName = imgDir + "/" + fileName;
		
		String out_fname = "";
		
		if( outDir.charAt(outDir.length()-1) == '/' )
			out_fname = outDir + fileName;
		else
			out_fname = outDir + "/" + fileName;
		
		
		// added on Feb. 19, 2014
		if( rectangles.size() >= 160 )
			return "0:::";
				
		
		ArrayList<UtilFunctions.BBoxCoor> numbers = new ArrayList<UtilFunctions.BBoxCoor>(); 
		
		for(i = 0; i<rectangles.size(); i+=4){
			UtilFunctions.BBoxCoor each_rect = new UtilFunctions.BBoxCoor();
			
			each_rect.left = rectangles.get(i);
			each_rect.top = rectangles.get(i+1);
			each_rect.right = rectangles.get(i+2);
			each_rect.bottom = rectangles.get(i+3);
			
			numbers.add(each_rect);		
		}
		
			
		IplImage inImg = cvLoadImage(imgName);
		
	//	cvSaveImage(out_fname, inImg);
		
		IplImage img, resized_img;
		ArrayList<PanelLabelDetection.PanelLabelInfo_Final> candLabelSet = new ArrayList<PanelLabelDetection.PanelLabelInfo_Final>();

		PanelLabelDetection.PanelLabelInfo_Final finalLabelSet = new PanelLabelDetection.PanelLabelInfo_Final();
		int num_split_panels = 0;
		
		height = inImg.height();
		width = inImg.width();
		
		// need to adjust the right and bottom coordinate.
		// the results include width and height as maximum coordinates, but functions allow only
		// up to width-1 and height-1 coordinates. 
		for (i = 0; i < numbers.size(); i++) {
			if (numbers.get(i).right >= width)
				numbers.get(i).right = width - 1;

			if (numbers.get(i).bottom >= height)
				numbers.get(i).bottom = height - 1;
		}

		// 200% enlargement for small image and color conversion
		resized_img = null;
		img = cvCreateImage(cvGetSize(inImg), IPL_DEPTH_8U, 1);

		if (inImg.nChannels() == 3 )
			cvCvtColor(inImg, img, CV_RGB2GRAY);			
		else
			cvCopy(inImg, img, null);

	//	Arrays.fill(candLabelSet, 0);
		
		PanelLabelDetection panelLabelDetectionRef = new PanelLabelDetection();
		
		
		// panel label detection. 
		// candLabelSet has all candidate label sets and the setNo denotes the total cand numbers. 
		if (width < 1000 && height < 1000) {
			resized_img = cvCreateImage(cvSize(width * 2, height * 2), IPL_DEPTH_8U, 1);
			cvResize(img, resized_img, 1);
			panelLabelDetectionRef.panelLabelDetection(resized_img, true, candLabelSet, OCR_model);			
		} else
			panelLabelDetectionRef.panelLabelDetection(img, false, candLabelSet, OCR_model);
		
		
		setNo = panelLabelDetectionRef.setNo;		
			

		{
			int max_match_set_idx = 0;
			int max_match_no, max_unmatch_no;
			
			int the_idx = 0;
			int pos_x, pos_y;
			int min_dist;
			int x_interval, y_interval, main_pos;
			
			int x_diff, y_diff;
			int left_margin, top_margin, right_margin, bottom_margin;

			int [] label_pos = new int [9];
			ArrayList<Integer> named_panel;
			ArrayList<Integer> split_box_idx;
			ArrayList<Integer> label_idx_in_one_box;

			int [] checked_label = new int [26];
			String panel_name = "";
			char the_label = 0;
			ArrayList<Character> named_panel_label = null;
			
			boolean new_panel_added, found_label;

			int no_already_split = 0;

			// indicate whether a panel in numbers array is labeled or not.  
			named_panel = new ArrayList<Integer>();
			split_box_idx = new ArrayList<Integer>();
			named_panel_label = new ArrayList<Character>();
			label_idx_in_one_box = new ArrayList<Integer>();
			
			for(i=0; i<numbers.size();i++){
				named_panel.add(0);
				split_box_idx.add(0);
				named_panel_label.add(' ');
			}
				
			// find a candidate that has the most identical labels with text labels.
			// just match text labels with OCR results. 

			max_match_no = max_unmatch_no = 0;
			///	max_case = 0;
			//	for(case_idx=0; case_idx < 1; case_idx++)
			if (no_text_panel > 0) {
				for (i = 0; i < setNo; i++) {
					match_label_no = unmatch_label_no = 0;
					Arrays.fill(checked_label, 0);
					for (j = 0; j < candLabelSet.get(i).noLabel; j++) {
						if (candLabelSet.get(i).labels.get(j).label == 0)
							continue;
												
						if( text_labels.indexOf(candLabelSet.get(i).labels.get(j).label) != -1 ){
						
							if (candLabelSet.get(i).labels.get(j).label >= 97) {
								if (checked_label[Math.min(candLabelSet.get(i).labels.get(j).label - 97, 25)] == 0) {
									match_label_no++;
									checked_label[Math.min(candLabelSet.get(i).labels.get(j).label - 97, 25)] = 1;
								}
							}

							if (candLabelSet.get(i).labels.get(j).label < 97 && candLabelSet.get(i).labels.get(j).label >= 65) {
								if (checked_label[Math.min(candLabelSet.get(i).labels.get(j).label - 65, 25)] == 0) {
									match_label_no++;
									checked_label[Math.min(candLabelSet.get(i).labels.get(j).label - 65, 25)] = 1;
								}
							}
						} else
							unmatch_label_no++;
					}

					if (match_label_no > max_match_no) {
						max_match_no = match_label_no;
						max_unmatch_no = unmatch_label_no;
						max_match_set_idx = i;
					}

					if (match_label_no == max_match_no && max_unmatch_no > unmatch_label_no) {
						max_unmatch_no = unmatch_label_no;
						max_match_set_idx = i;
					}
				}
			}

			Arrays.fill(checked_label, 0);

			// found a cand set that has the most identical labels with text labels. 
			if (max_match_no > 0) {

				// eliminate noise labels (unmatched labels) from the chosen candidate set. 
								
				if (candLabelSet.get(max_match_set_idx).noLabel > no_text_panel || candLabelSet.get(max_match_set_idx).noLabel > max_match_no) {
					for (j = 0; j < candLabelSet.get(max_match_set_idx).noLabel; j++) {

						if (candLabelSet.get(max_match_set_idx).labels.get(j).label == 0)
							continue;
						
						if ( text_labels.indexOf(candLabelSet.get(max_match_set_idx).labels.get(j).label) == -1 )
							candLabelSet.get(max_match_set_idx).labels.get(j).label = 0;
						else 
						{
							//	*ch_str = 95;
							for (i = 0; i < candLabelSet.get(max_match_set_idx).noLabel; i++) {
								if (i == j || candLabelSet.get(max_match_set_idx).labels.get(i).label == 0)
									continue;

								// in case the set has multiple identical labels, then pick a label with largest msg_1.
								if (candLabelSet.get(max_match_set_idx).labels.get(i).label == candLabelSet.get(max_match_set_idx).labels.get(j).label) {
									if (candLabelSet.get(max_match_set_idx).labels.get(i).msg_1 > candLabelSet.get(max_match_set_idx).labels.get(j).msg_1) {
										candLabelSet.get(max_match_set_idx).labels.get(j).label = 0;
										break;
									} else
										candLabelSet.get(max_match_set_idx).labels.get(i).label = 0;
								}
							}
						}
					}
				}

				finalLabelSet.upper_or_lower_case = candLabelSet.get(max_match_set_idx).upper_or_lower_case;
				finalLabelSet.labels = new ArrayList<PanelLabelDetection.PanelLabelInfo>();
				
				for (i = 0; i < candLabelSet.get(max_match_set_idx).noLabel; i++) {
					if (candLabelSet.get(max_match_set_idx).labels.get(i).label == 0)
						continue;
					
					PanelLabelDetection.PanelLabelInfo tmp_Info = new PanelLabelDetection.PanelLabelInfo();
					tmp_Info = candLabelSet.get(max_match_set_idx).labels.get(i);
					
					finalLabelSet.labels.add(tmp_Info);				
					
				//	System.arraycopy(candLabelSet.get(max_match_set_idx).labels, i, finalLabelSet.labels, finalLabelSet.noLabel, 1);
				//	finalLabelSet.noLabel++;
				}

				finalLabelSet.noLabel = finalLabelSet.labels.size();
				// count valid labels that can be used for labeling.
				
				for (j = 0; j < finalLabelSet.noLabel; j++) {
					if ( finalLabelSet.labels.get(j).label == 0 )
						continue;
				}
				
				
				// panel splitting with multiple panel labels in it.
				
				new_panel_added = true;

				// for left to right arrangement only.
				// need to implement top to bottom arrangement. 
				//	if( no_panel == 1 && no_valid_labels > 1 )
				{
					int init_num = numbers.size();
					
					while( new_panel_added == true ) {

						new_panel_added = false;
						for (i = 0; i < Math.min(init_num*2, numbers.size()); i++) {
							label_idx_in_one_box.clear();
							
							for (j = 0; j < finalLabelSet.noLabel; j++) {
								if (finalLabelSet.labels.get(j).label == 0)
									continue;

								if (finalLabelSet.labels.get(j).cen_x > numbers.get(i).left && finalLabelSet.labels.get(j).cen_y > numbers.get(i).top && finalLabelSet.labels.get(j).cen_x < numbers.get(i).right && finalLabelSet.labels.get(j).cen_y < numbers.get(i).bottom) {
									label_idx_in_one_box.add(j);									
								}
							}

							if (label_idx_in_one_box.size() > 2) {

								// sort the OCRed labels.
								idx = 0;
								Arrays.fill(label_order, 0);
								for (j = 0; j < label_idx_in_one_box.size(); j++) {

									label_order[idx] = finalLabelSet.labels.get(label_idx_in_one_box.get(j)).bottom;
									ori_label_idx[idx++] = j;
									
									if( idx > 25 )
										idx = 25;
								}

								utilFunc.sorting_min_2_max(label_order, sorted_label_idx, idx);

								left_margin = top_margin = right_margin = bottom_margin = 1000;
								for (j = 0; j < label_idx_in_one_box.size(); j++) {
									left_margin = Math.min(finalLabelSet.labels.get(label_idx_in_one_box.get(j)).left - numbers.get(i).left, left_margin);
									top_margin = Math.min(finalLabelSet.labels.get(label_idx_in_one_box.get(j)).top - numbers.get(i).top, top_margin);
									right_margin = Math.min(numbers.get(i).right - finalLabelSet.labels.get(label_idx_in_one_box.get(j)).right, right_margin);
									bottom_margin = Math.min(numbers.get(i).bottom - finalLabelSet.labels.get(label_idx_in_one_box.get(j)).bottom, bottom_margin);
								}

								// bottom_left (right) label pattern. 
								if (bottom_margin < top_margin && bottom_margin < 100 && ((left_margin < right_margin && left_margin < 100) || (right_margin < left_margin && right_margin < 100))) {

									for (j = label_idx_in_one_box.size() - 1; j > 0; j--) {
										if (Math.abs(finalLabelSet.labels.get(label_idx_in_one_box.get(ori_label_idx[sorted_label_idx[j]])).cen_y - finalLabelSet.labels.get(label_idx_in_one_box.get(ori_label_idx[sorted_label_idx[j - 1]])).cen_y) > 100) {
											
											UtilFunctions.BBoxCoor tmp_coor = new UtilFunctions.BBoxCoor();
											
											tmp_coor.left = numbers.get(i).left;
											tmp_coor.top = finalLabelSet.labels.get(label_idx_in_one_box.get(ori_label_idx[sorted_label_idx[j - 1]])).bottom + bottom_margin;
											tmp_coor.right = numbers.get(i).right;
											tmp_coor.bottom = numbers.get(i).bottom; 
											
											numbers.get(i).bottom = tmp_coor.top - 1;
											
											numbers.add(tmp_coor);
											named_panel.add(0);
											split_box_idx.add(0);
											named_panel_label.add(' ');
											new_panel_added = true;
										}
									}
								}

								// top_left (right) label pattern. 
								if (top_margin < bottom_margin && top_margin < 100
										&& ((left_margin < right_margin
												&& left_margin < 100)
												|| (right_margin < left_margin
														&& right_margin < 100))) {

									for (j = label_idx_in_one_box.size() - 1; j > 0; j--) {
										if (Math.abs(finalLabelSet.labels.get(label_idx_in_one_box.get(ori_label_idx[sorted_label_idx[j]])).cen_y - finalLabelSet.labels.get(label_idx_in_one_box.get(ori_label_idx[sorted_label_idx[j - 1]])).cen_y) > 100) {
											UtilFunctions.BBoxCoor tmp_coor = new UtilFunctions.BBoxCoor();
											
											tmp_coor.left = numbers.get(i).left;
											tmp_coor.top = finalLabelSet.labels.get(label_idx_in_one_box.get(ori_label_idx[sorted_label_idx[j]])).top - top_margin;
											tmp_coor.right = numbers.get(i).right; 
											tmp_coor.bottom = numbers.get(i).bottom;
											
											numbers.get(i).bottom = tmp_coor.top - 1;
											
											numbers.add(tmp_coor);
											named_panel.add(0);
											split_box_idx.add(0);
											named_panel_label.add(' ');
											
											new_panel_added = true;											
										}
									}
								}
							}
						}					
					}
				}
				//	else

				{
					new_panel_added = true;
					int init_num = numbers.size();
					
					while (new_panel_added == true ) {

						new_panel_added = false;

						for (i = 0; i < Math.min(init_num*2, numbers.size()); i++) {
							
							label_idx_in_one_box.clear();

							for (j = 0; j < finalLabelSet.noLabel; j++) {
								if (finalLabelSet.labels.get(j).label == 0)
									continue;

								if (finalLabelSet.labels.get(j).cen_x > numbers.get(i).left && finalLabelSet.labels.get(j).cen_y > numbers.get(i).top && finalLabelSet.labels.get(j).cen_x < numbers.get(i).right && finalLabelSet.labels.get(j).cen_y < numbers.get(i).bottom) {
									label_idx_in_one_box.add(j);
								}
							}

							// two labels in one box
							if (label_idx_in_one_box.size() >= 2) {
								// find the arrangement of the labels. compare the x and y coordinates of the center.
								// if x_diff is very small, then vertical arrangement. 
								// if y_diff is very small, then horizontal arrangement. 

								// find where the labels are located. Usually on a corner of the rectangle. 
								// if conditions met, then split it into two and store the new rectangle and modify the original one. 
								x_diff = Math.abs(finalLabelSet.labels.get(label_idx_in_one_box.get(0)).cen_x - finalLabelSet.labels.get(label_idx_in_one_box.get(1)).cen_x);
								y_diff = Math.abs(finalLabelSet.labels.get(label_idx_in_one_box.get(0)).cen_y - finalLabelSet.labels.get(label_idx_in_one_box.get(1)).cen_y);

								// left to right arrangement
								if (y_diff < finalLabelSet.labels.get(label_idx_in_one_box.get(0)).bottom - finalLabelSet.labels.get(label_idx_in_one_box.get(0)).top) {

									idx = 0;
									Arrays.fill(label_order, 0);
									Arrays.fill(ori_label_idx, 0);
									Arrays.fill(sorted_label_idx, 0);
									
									for (j = 0; j < label_idx_in_one_box.size(); j++) {
										label_order[idx] = finalLabelSet.labels.get(label_idx_in_one_box.get(j)).right;
										ori_label_idx[idx++] = j;
										
										if( idx > 25 )
											idx = 25;
									}

									utilFunc.sorting_max_2_min_INT(label_order, sorted_label_idx, idx);

									left_margin = right_margin = 1000;
									for (j = 0; j < label_idx_in_one_box.size(); j++) {
										left_margin = Math.min(left_margin, finalLabelSet.labels.get(label_idx_in_one_box.get(j)).left - numbers.get(i).left);
										right_margin = Math.min(right_margin, numbers.get(i).right - finalLabelSet.labels.get(label_idx_in_one_box.get(j)).right);
									}

									// left corner location
									if (left_margin < right_margin && left_margin < 100) {
										for (j = 0; j < label_idx_in_one_box.size() - 1; j++) {
											UtilFunctions.BBoxCoor tmp_coor = new UtilFunctions.BBoxCoor();
											
											tmp_coor.left = finalLabelSet.labels.get(label_idx_in_one_box.get(ori_label_idx[sorted_label_idx[j]])).left - left_margin;
											tmp_coor.top = numbers.get(i).top;
											tmp_coor.right = numbers.get(i).right;
											tmp_coor.bottom = numbers.get(i).bottom;
											
											numbers.get(i).right = tmp_coor.left - 1;
											
											numbers.add(tmp_coor);
											named_panel.add(0);
											split_box_idx.add(0);
											named_panel_label.add(' ');
											
										}
									} else
									// right corner location
									if (right_margin < left_margin && right_margin < 100) {

										for (j = 1; j < label_idx_in_one_box.size(); j++) {
											UtilFunctions.BBoxCoor tmp_coor = new UtilFunctions.BBoxCoor();
											tmp_coor.left = finalLabelSet.labels.get(label_idx_in_one_box.get(ori_label_idx[sorted_label_idx[j]])).right + right_margin;
											tmp_coor.top = numbers.get(i).top;
											tmp_coor.right = numbers.get(i).right;
											tmp_coor.bottom = numbers.get(i).bottom;
											
											numbers.get(i).right = tmp_coor.left - 1;
											
											numbers.add(tmp_coor);
											named_panel.add(0);
											split_box_idx.add(0);
											named_panel_label.add(' ');
										}
									}
								}

								// top to bottom arrangement
								if (x_diff < finalLabelSet.labels.get(label_idx_in_one_box.get(0)).right - finalLabelSet.labels.get(label_idx_in_one_box.get(0)).left) {

									idx = 0;
									Arrays.fill(label_order, 0);
									Arrays.fill(ori_label_idx, 0);
									Arrays.fill(sorted_label_idx, 0);
									
									for (j = 0; j < label_idx_in_one_box.size(); j++) {

										label_order[idx] = finalLabelSet.labels.get(label_idx_in_one_box.get(j)).bottom;
										ori_label_idx[idx++] = j;
										
										if( idx > 25 )
											idx = 25;
									}

									utilFunc.sorting_max_2_min_INT(label_order, sorted_label_idx, idx);

									top_margin = bottom_margin = 1000;
									for (j = 0; j < label_idx_in_one_box.size(); j++) {
										top_margin = Math.min(top_margin, finalLabelSet.labels.get(label_idx_in_one_box.get(j)).top - numbers.get(i).top);
										bottom_margin = Math.min(bottom_margin, numbers.get(i).bottom - finalLabelSet.labels.get(label_idx_in_one_box.get(j)).bottom);
									}

									// upper corner location.
									if (top_margin < bottom_margin
											&& top_margin < 100) {

										for (j = 0; j < label_idx_in_one_box.size() - 1;	j++) {
											UtilFunctions.BBoxCoor tmp_coor = new UtilFunctions.BBoxCoor();
											
											tmp_coor.left = numbers.get(i).left;
											tmp_coor.top = finalLabelSet.labels.get(label_idx_in_one_box.get(ori_label_idx[sorted_label_idx[j]])).top - top_margin;
											tmp_coor.right = numbers.get(i).right;
											tmp_coor.bottom = numbers.get(i).bottom;
											
											numbers.get(i).bottom = tmp_coor.top - 1;
											
											numbers.add(tmp_coor);		
											named_panel.add(0);
											split_box_idx.add(0);
											named_panel_label.add(' ');
										}
									}

									// lower corner location
									if (bottom_margin < top_margin && bottom_margin < 100) {

										for (j = 1; j < label_idx_in_one_box.size(); j++) {
											
											UtilFunctions.BBoxCoor tmp_coor = new UtilFunctions.BBoxCoor();
											tmp_coor.left = numbers.get(i).left;
											tmp_coor.top = finalLabelSet.labels.get(label_idx_in_one_box.get(ori_label_idx[sorted_label_idx[j]])).bottom + bottom_margin;
											tmp_coor.right = numbers.get(i).right;
											tmp_coor.bottom = numbers.get(i).bottom;
											
											numbers.get(i).bottom = tmp_coor.top-1;
											
											numbers.add(tmp_coor);
											named_panel.add(0);
											split_box_idx.add(0);
											named_panel_label.add(' ');
										}
									}
								}
							}
						}					
					}
				}

				// try to label each panel that contains only one label in it. 
				Arrays.fill(label_pos, 0);
				
								
				for (i = 0; i < numbers.size(); i++) {
					int no_labels_in_a_box = 0;

					for (j = 0; j < finalLabelSet.noLabel; j++) {
						if (finalLabelSet.labels.get(j).label == 0)
							continue;

						if (finalLabelSet.labels.get(j).cen_x > numbers.get(i).left && finalLabelSet.labels.get(j).cen_y > numbers.get(i).top && finalLabelSet.labels.get(j).cen_x < numbers.get(i).right && finalLabelSet.labels.get(j).cen_y < numbers.get(i).bottom) {

							the_label = finalLabelSet.labels.get(j).label;
							the_idx = j;
							no_labels_in_a_box++;
						}
					}

					// there's a only one label in the box
					// split and label it. 
					if (no_labels_in_a_box == 1) {

						if (numbers.get(i).left >= width || numbers.get(i).right >= width
								|| numbers.get(i).top >= height
								|| numbers.get(i).bottom >= height
								|| numbers.get(i).right <= numbers.get(i).left
								|| numbers.get(i).bottom <= numbers.get(i).top)
							continue;

						if (split_box_idx.get(i) > 0)
							continue;

						cvSetImageROI(inImg, cvRect(numbers.get(i).left,numbers.get(i).top, Math.min(numbers.get(i).right - numbers.get(i).left + 1, inImg.width() - 1), Math.min(numbers.get(i).bottom - numbers.get(i).top + 1, inImg.height() - 1)));
						IplImage cropped = cvCreateImage(cvSize(Math.min(numbers.get(i).right - numbers.get(i).left + 1, inImg.width() - 1), Math.min(numbers.get(i).bottom - numbers.get(i).top + 1, inImg.height() - 1)), inImg.depth(), inImg.nChannels());
						cvCopy(inImg, cropped, null);

						panel_name = "";
						panel_name = out_fname;
						
						panel_name = panel_name.split(".jpg")[0] + '_' + the_label + ".jpg";

						// if statement added on 12/19/2012
						if( alphaStr.indexOf(the_label) != -1 ){
							cvSaveImage(panel_name, cropped);
							final_label_list += the_label;
							num_split_panels++;
						//	printf("save 1: %c\n", the_label);
							
							Final_Panel tmp_panel = new Final_Panel();
							tmp_panel.left = numbers.get(i).left;
							tmp_panel.top = numbers.get(i).top;
							tmp_panel.right = numbers.get(i).right;
							tmp_panel.bottom = numbers.get(i).bottom;
							tmp_panel.label = the_label;
							
							finalSplitPanels.add(tmp_panel);
						}

						cvReleaseImage(cropped); cropped = null;
						cvResetImageROI(inImg);

						finalLabelSet.labels.get(the_idx).used = true;
						named_panel.set(i, 1);
						named_panel_label.set(i, finalLabelSet.labels.get(the_idx).label);
						split_box_idx.set(i, 1);
						no_already_split++;

						if( named_panel_label.get(i) >= 97 )
							checked_label[Math.min(named_panel_label.get(i) - 97, 25)] = 1;
						else
							if( named_panel_label.get(i) < 97 && named_panel_label.get(i) >= 65 )
								checked_label[Math.min(named_panel_label.get(i) - 65, 25)] = 1;
						
						
						// find the pattern of location of labels. 3*3 position mask. 
						if (finalLabelSet.labels.get(the_idx).cen_x < (numbers.get(i).right - numbers.get(i).left) / 3 + numbers.get(i).left)
							pos_x = 0;
						else if (finalLabelSet.labels.get(the_idx).cen_x > numbers.get(i).right - (numbers.get(i).right - numbers.get(i).left) / 3)
							pos_x = 2;
						else
							pos_x = 1;

						if (finalLabelSet.labels.get(the_idx).cen_y < (numbers.get(i).bottom - numbers.get(i).top) / 3 + numbers.get(i).top)
							pos_y = 0;
						else if (finalLabelSet.labels.get(the_idx).cen_y > numbers.get(i).bottom - (numbers.get(i).bottom - numbers.get(i).top) / 3)
							pos_y = 2;
						else
							pos_y = 1;

						label_pos[pos_y * 3 + pos_x]++;
					}
				}

				k = 0;
				main_pos = -1;
				for (i = 0; i < 9; i++) {
					if (label_pos[i] > 0) {
						main_pos = i;
						k++;
					}
				}

				// for now, only think about one label location pattern, e.g., all labels are located on the upper-left corner.
				// otherwise, don't use this information. 
				if (k > 1)
					main_pos = -1;

				// process if only one label position pattern detected. 
				// try to find labels out of the panels. 
				// assume that we just move the bounding box a little bit to contain a certain label near the box and satisfy
				// the pattern. The label is right label for the panel. 
				if (main_pos != -1) {
					for (i = 0; i < numbers.size(); i++) {
						if (named_panel.get(i) > 0)
							continue;

						found_label = false;
						min_dist = 10000;

						x_interval = (numbers.get(i).right - numbers.get(i).left) / 3;
						y_interval = (numbers.get(i).bottom - numbers.get(i).top) / 3;

						for (j = 0; j < finalLabelSet.noLabel; j++) {

							if (finalLabelSet.labels.get(j).used == true || finalLabelSet.labels.get(j).label == 0 )
								continue;

							switch (main_pos) {
							// upper left corner. the top or left can be moved to contain the label.
							// find the closest one.
							case 0:
								if (finalLabelSet.labels.get(j).cen_y > numbers.get(i).top && finalLabelSet.labels.get(j).cen_y < numbers.get(i).top + y_interval) {

									if (finalLabelSet.labels.get(j).cen_x < numbers.get(i).left && numbers.get(i).left - finalLabelSet.labels.get(j).cen_x < min_dist) {
										min_dist = numbers.get(i).left - finalLabelSet.labels.get(j).cen_x;
										the_idx = j;
										the_label = finalLabelSet.labels.get(j).label;

										found_label = true;
									}
								}

								if (finalLabelSet.labels.get(j).cen_x > numbers.get(i).left && finalLabelSet.labels.get(j).cen_x < numbers.get(i).left + x_interval) {

									if (finalLabelSet.labels.get(j).cen_y < numbers.get(i).top && numbers.get(i).top - finalLabelSet.labels.get(j).cen_y < min_dist) {
										min_dist = numbers.get(i).top - finalLabelSet.labels.get(j).cen_y;
										the_idx = j;
										the_label = finalLabelSet.labels.get(j).label;

										found_label = true;
									}
								}

								break;

								// lower-left corner.
							case 6:
								if (finalLabelSet.labels.get(j).cen_y < numbers.get(i).bottom && finalLabelSet.labels.get(j).cen_y > numbers.get(i).bottom - y_interval) {

									if (finalLabelSet.labels.get(j).cen_x < numbers.get(i).left && numbers.get(i).left - finalLabelSet.labels.get(j).cen_x < min_dist) {
										min_dist = numbers.get(i).left - finalLabelSet.labels.get(j).cen_x;
										the_idx = j;
										the_label = finalLabelSet.labels.get(j).label;

										found_label = true;
									}
								}

								if (finalLabelSet.labels.get(j).cen_x > numbers.get(i).left && finalLabelSet.labels.get(j).cen_x < numbers.get(i).left + x_interval) {

									if (finalLabelSet.labels.get(j).cen_y > numbers.get(i).bottom && finalLabelSet.labels.get(j).cen_y - numbers.get(i).bottom < min_dist) {
										min_dist = finalLabelSet.labels.get(j).cen_y - numbers.get(i).bottom;
										the_idx = j;
										the_label = finalLabelSet.labels.get(j).label;

										found_label = true;
									}
								}

								break;

							default:
								break;
							}
						}

						// if a label is found for this rectangle, then name the panel. 
						if (found_label == true ) {

							if (numbers.get(i).left >= width
									|| numbers.get(i).right >= width
									|| numbers.get(i).top >= height
									|| numbers.get(i).bottom >= height
									|| numbers.get(i).right <= numbers.get(i).left
									|| numbers.get(i).bottom <= numbers.get(i).top)
								continue;

							if (split_box_idx.get(i) > 0)
								continue;

							cvSetImageROI(inImg, cvRect(numbers.get(i).left,numbers.get(i).top,Math.min(numbers.get(i).right- numbers.get(i).left + 1,inImg.width() - 1),Math.min(numbers.get(i).bottom- numbers.get(i).top + 1,inImg.height() - 1)));
							IplImage cropped = cvCreateImage(cvSize(Math.min(numbers.get(i).right - numbers.get(i).left + 1,inImg.width() - 1),Math.min(numbers.get(i).bottom - numbers.get(i).top + 1, inImg.height() - 1)),inImg.depth(), inImg.nChannels());
							cvCopy(inImg, cropped, null);

							panel_name = "";
							panel_name = out_fname;
														
							panel_name = panel_name.split(".jpg")[0] + '_' + the_label + ".jpg";
							
							if( alphaStr.indexOf(the_label) != -1 ){
								cvSaveImage(panel_name, cropped);
								final_label_list += the_label;
								num_split_panels++;
							//	printf("save 2: %c\n", the_label);
								
								Final_Panel tmp_panel = new Final_Panel();
								tmp_panel.left = numbers.get(i).left;
								tmp_panel.top = numbers.get(i).top;
								tmp_panel.right = numbers.get(i).right;
								tmp_panel.bottom = numbers.get(i).bottom;
								tmp_panel.label = the_label;
								
								finalSplitPanels.add(tmp_panel);
							}


							cvReleaseImage(cropped); cropped = null;
							cvResetImageROI(inImg);

							finalLabelSet.labels.get(the_idx).used = true;
							named_panel.set(i, 1);
							named_panel_label.set(i, finalLabelSet.labels.get(the_idx).label);
							split_box_idx.set(i, 1);
							no_already_split++;

							if( named_panel_label.get(i) >= 97 )
								checked_label[Math.min(named_panel_label.get(i)-97, 25)] = 1;
							else
								if( named_panel_label.get(i) < 97 && named_panel_label.get(i) >= 65 )
									checked_label[Math.min(named_panel_label.get(i)-65, 25)] = 1;
						}
					}
				}
			}


			// updated...
			// no panels are labeled.
			// panel labels are outside of panels. 
			// cause problems with awp328f7
			// applied only no panels are labeled before since all labels are outside of panels. 

			if (no_already_split == 0) {
				int type_idx;
				char [][] named_panel_label_tmp = new char [6][50];
				int [][] named_panel_label_tmp_idx = new int [6][50];
				int [] assigned_label_no = new int [6];
				
			//	Arrays.fill(named_panel_label_tmp, 0);
			//	Arrays.fill(named_panel_label_tmp_idx, 0);
			//	Arrays.fill(assigned_label_no, 0);
				
				for (type_idx = 0; type_idx < 6; type_idx++) {

					switch (type_idx) {

					case 0:
						for (i = 0; i < Math.min(numbers.size(), 50); i++) {

							if (named_panel.get(i) > 0)
								continue;

							for (j = 0; j < finalLabelSet.noLabel; j++) {
								if (finalLabelSet.labels.get(j).label == 0 || finalLabelSet.labels.get(j).used == true )
									continue;

								if (finalLabelSet.labels.get(j).cen_x > numbers.get(i).left && finalLabelSet.labels.get(j).cen_y < numbers.get(i).top && finalLabelSet.labels.get(j).cen_x < numbers.get(i).left + (numbers.get(i).right - numbers.get(i).left) / 3 && numbers.get(i).top - finalLabelSet.labels.get(j).cen_y < 100) {
									named_panel_label_tmp[type_idx][i] = finalLabelSet.labels.get(j).label;
									named_panel_label_tmp_idx[type_idx][i] = j;
									assigned_label_no[type_idx]++;
									break;
								}
							}
						}

						break;

					case 1:

						for (i = 0; i < Math.min(numbers.size(), 50); i++) {

							if (named_panel.get(i) > 0)
								continue;

							for (j = 0; j < finalLabelSet.noLabel; j++) {
								if (finalLabelSet.labels.get(j).label == 0 || finalLabelSet.labels.get(j).used == true )
									continue;

								if (finalLabelSet.labels.get(j).cen_x < numbers.get(i).left && finalLabelSet.labels.get(j).cen_y > numbers.get(i).top && finalLabelSet.labels.get(j).cen_y < numbers.get(i).top + (numbers.get(i).bottom - numbers.get(i).top) / 3 && numbers.get(i).left - finalLabelSet.labels.get(j).cen_x < 100) {
									named_panel_label_tmp[type_idx][i] = finalLabelSet.labels.get(j).label;
									named_panel_label_tmp_idx[type_idx][i] = j;
									assigned_label_no[type_idx]++;
									break;
								}
							}
						}

						break;

					case 2:

						for (i = 0; i < Math.min(numbers.size(), 50); i++) {

							if (named_panel.get(i) > 0)
								continue;

							for (j = 0; j < finalLabelSet.noLabel; j++) {
								if (finalLabelSet.labels.get(j).label == 0 || finalLabelSet.labels.get(j).used == true)
									continue;

								if (finalLabelSet.labels.get(j).cen_x < numbers.get(i).left && finalLabelSet.labels.get(j).cen_y < numbers.get(i).bottom && finalLabelSet.labels.get(j).cen_y > numbers.get(i).bottom - (numbers.get(i).bottom - numbers.get(i).top) / 3 && numbers.get(i).left - finalLabelSet.labels.get(j).cen_x < 100) {
									named_panel_label_tmp[type_idx][i] = finalLabelSet.labels.get(j).label;
									named_panel_label_tmp_idx[type_idx][i] = j;
									assigned_label_no[type_idx]++;
									break;
								}
							}
						}

						break;

					case 3:

						for (i = 0; i < Math.min(numbers.size(),50); i++) {

							if (named_panel.get(i) > 0)
								continue;

							for (j = 0; j < finalLabelSet.noLabel; j++) {
								if (finalLabelSet.labels.get(j).label == 0 || finalLabelSet.labels.get(j).used == true )
									continue;

								if (finalLabelSet.labels.get(j).cen_x > numbers.get(i).left && finalLabelSet.labels.get(j).cen_y > numbers.get(i).bottom && finalLabelSet.labels.get(j).cen_x < numbers.get(i).left + (numbers.get(i).right - numbers.get(i).left) / 3 && finalLabelSet.labels.get(j).cen_y - numbers.get(i).bottom < 100) {
									named_panel_label_tmp[type_idx][i] = finalLabelSet.labels.get(j).label;
									named_panel_label_tmp_idx[type_idx][i] = j;
									assigned_label_no[type_idx]++;
									break;
								}
							}
						}

						break;

					case 4:

						for (i = 0; i < Math.min(numbers.size(), 50); i++) {

							if (named_panel.get(i) > 0)
								continue;

							for (j = 0; j < finalLabelSet.noLabel; j++) {
								if (finalLabelSet.labels.get(j).label == 0 || finalLabelSet.labels.get(j).used == true )
									continue;

								if (finalLabelSet.labels.get(j).cen_x < numbers.get(i).left && finalLabelSet.labels.get(j).cen_x > numbers.get(i).left - 50 && finalLabelSet.labels.get(j).cen_y > numbers.get(i).top - 50 && finalLabelSet.labels.get(j).cen_y < numbers.get(i).top + 50) {

									named_panel_label_tmp[type_idx][i] = finalLabelSet.labels.get(j).label;
									named_panel_label_tmp_idx[type_idx][i] = j;
									assigned_label_no[type_idx]++;
									break;
								}
							}
						}

						break;

					case 5: // out, bottom, center of the panel.

						for (i = 0; i < Math.min(numbers.size(),50); i++) {

							if (named_panel.get(i) > 0)
								continue;

							for (j = 0; j < finalLabelSet.noLabel; j++) {
								if (finalLabelSet.labels.get(j).label == 0 || finalLabelSet.labels.get(j).used == true )
									continue;

								if (finalLabelSet.labels.get(j).cen_x > numbers.get(i).left + (numbers.get(i).right - numbers.get(i).left) / 3 && finalLabelSet.labels.get(j).cen_x < numbers.get(i).right - (numbers.get(i).right - numbers.get(i).left) / 3 && finalLabelSet.labels.get(j).cen_y - numbers.get(i).bottom < 100 && finalLabelSet.labels.get(j).cen_y - numbers.get(i).bottom > 0) {

									named_panel_label_tmp[type_idx][i] = finalLabelSet.labels.get(j).label;
									named_panel_label_tmp_idx[type_idx][i] = j;
									assigned_label_no[type_idx]++;
									break;
								}
							}
						}

						break;

					}
				}

				i = 0;
				j = 0;

				for (type_idx = 0; type_idx < 6; type_idx++) {
					if (assigned_label_no[type_idx] > i) {
						i = assigned_label_no[type_idx];
						j = type_idx;
					}
				}

				if (i > 0) {
					type_idx = j;

					for (i = 0; i < Math.min(numbers.size(),50); i++) {
						if (named_panel.get(i) > 0
								|| named_panel_label_tmp[type_idx][i] == 0
								|| split_box_idx.get(i) > 0)
							continue;

						if (numbers.get(i).left >= width || numbers.get(i).right >= width
								|| numbers.get(i).top >= height
								|| numbers.get(i).bottom >= height
								|| numbers.get(i).right <= numbers.get(i).left
								|| numbers.get(i).bottom <= numbers.get(i).top)
							continue;

						cvSetImageROI(inImg, cvRect(numbers.get(i).left,numbers.get(i).top,Math.min(numbers.get(i).right - numbers.get(i).left+ 1, inImg.width() - 1),Math.min(numbers.get(i).bottom- numbers.get(i).top + 1,inImg.height() - 1)));
						
						IplImage cropped = cvCreateImage(cvSize(Math.min(numbers.get(i).right - numbers.get(i).left+ 1, inImg.width() - 1),Math.min(numbers.get(i).bottom- numbers.get(i).top + 1,inImg.height() - 1)), inImg.depth(),inImg.nChannels());
						cvCopy(inImg, cropped, null);

						panel_name = "";
						panel_name = out_fname;
						panel_name = panel_name.split(".jpg")[0] + '_' + named_panel_label_tmp[type_idx][i] + ".jpg";

						if( alphaStr.indexOf(named_panel_label_tmp[type_idx][i]) != -1 ){
							cvSaveImage(panel_name, cropped);
							final_label_list += named_panel_label_tmp[type_idx][i];
							num_split_panels++;
						//	printf("save 3: %c\n", named_panel_label_tmp[type_idx][i]);
							
							Final_Panel tmp_panel = new Final_Panel();
							tmp_panel.left = numbers.get(i).left;
							tmp_panel.top = numbers.get(i).top;
							tmp_panel.right = numbers.get(i).right;
							tmp_panel.bottom = numbers.get(i).bottom;
							tmp_panel.label = named_panel_label_tmp[type_idx][i];
							
							finalSplitPanels.add(tmp_panel);							
						}

						cvReleaseImage(cropped); cropped = null;
						cvResetImageROI(inImg);
						
						finalLabelSet.labels.get(named_panel_label_tmp_idx[type_idx][i]).used = true;
						named_panel.set(i, 1);
						named_panel_label.set(i, named_panel_label_tmp[type_idx][i]);
						split_box_idx.set(i, 1);
						no_already_split++;

						if( named_panel_label.get(i) >= 97 )
							checked_label[Math.min(named_panel_label.get(i)-97, 25)] = 1;
						else
							if( named_panel_label.get(i) < 97 && named_panel_label.get(i) >= 65 )
								checked_label[Math.min(named_panel_label.get(i)-65, 25)] = 1;
						
					}
				}

			}


			// so far two cases are processed. 
			// 1. panels containing single label.
			// 2. panels that can contain a label if the box is expanded a little bit. 
			// 3. all labels are outside of panels. 

			// Try to assign a label based on the panels with labels and the position infomation of all the detected labels and panels.

			{
				int tmp_dist, min_box_dist;
				int cur_idx = 0;
				int left_top_box_idx = 0, first_col_box_idx = 0;
				int [] box_order_idx = new int [numbers.size()];
				int [] checked_panel_idx = new int [numbers.size()];

				char first_named_label;

				char prev_box_label;

				boolean left_to_right_pattern, top_to_bottom_pattern;
				boolean continue_while;

				int min_hor_gap, min_hor_gap_idx;

				// store the indices of rectangles by left to right and top to bottom orders.
				
				Arrays.fill(box_order_idx, -1);
				Arrays.fill(checked_panel_idx, 0);

				left_to_right_pattern = top_to_bottom_pattern = false;
				// some rectangles don't have labels.
				// need to assign one

				j = 0;
				for (i = 0; i < numbers.size(); i++) {
					if (named_panel.get(i) > 0)
						j++;
				}

				//	if( no_valid_labels < no_panel ){

				// some panels don't have labels. 
				if (j < numbers.size() ) {

					// find the left-top box

					// find the top-left most box. 
					min_box_dist = 1000;
					for (i = 0; i < numbers.size(); i++) {
						tmp_dist = (int) Math.sqrt( (double) (numbers.get(i).left * numbers.get(i).left + numbers.get(i).top * numbers.get(i).top));

						if (tmp_dist < min_box_dist) {
							left_top_box_idx = i;
							min_box_dist = tmp_dist;
						}
					}

					// sort the OCRed labels.
					idx = 0;
					Arrays.fill(label_order, 0);
					Arrays.fill(ori_label_idx, 0);

					for (i = 0; i < finalLabelSet.noLabel; i++) {
						if (finalLabelSet.labels.get(i).label == 0)
							continue;

						label_order[idx] = finalLabelSet.labels.get(i).label;
						ori_label_idx[idx++] = i;
						
						if( idx > 25 )
							idx = 25;
					}

					utilFunc.sorting_min_2_max(label_order, sorted_label_idx, idx);

					// find the arrangement pattern. 
					for (i = 0; i < idx - 1; i++) {
						if (finalLabelSet.labels.get(ori_label_idx[sorted_label_idx[i + 1]]).label - finalLabelSet.labels.get(ori_label_idx[sorted_label_idx[i]]).label != 1)
							continue;

						x_diff = Math.abs(finalLabelSet.labels.get(ori_label_idx[sorted_label_idx[i]]).cen_x - finalLabelSet.labels.get(ori_label_idx[sorted_label_idx[i + 1]]).cen_x);
						y_diff = Math.abs(finalLabelSet.labels.get(ori_label_idx[sorted_label_idx[i]]).cen_y - finalLabelSet.labels.get(ori_label_idx[sorted_label_idx[i + 1]]).cen_y);

						if (y_diff < finalLabelSet.labels.get(ori_label_idx[sorted_label_idx[i]]).bottom - finalLabelSet.labels.get(ori_label_idx[sorted_label_idx[i]]).top) 
							left_to_right_pattern = true;

						if (x_diff < finalLabelSet.labels.get(ori_label_idx[sorted_label_idx[i]]).right - finalLabelSet.labels.get(ori_label_idx[sorted_label_idx[i]]).left)
							top_to_bottom_pattern = true;

						if (left_to_right_pattern == true || top_to_bottom_pattern == true )
							break;
					}

					//	if( left_to_right_pattern )
					{

						// find the first OCRed label. Most probably A or B (C, etc...)
						first_named_label = 127;
						for (i = 0; i < numbers.size(); i++) {
							if (named_panel.get(i) > 0 && named_panel_label.get(i) < first_named_label) {
								first_named_label = named_panel_label.get(i);
							}
						}

						cur_idx = left_top_box_idx;
						continue_while = true;
						checked_panel_idx[cur_idx] = 1;
						box_order_idx[0] = left_top_box_idx;
						idx = 1;

						// try to find rectangle indices closet to current box. 
						// find the order of the boxes from left to right, top to bottom.
						while (continue_while == true ) {

							continue_while = false;
							min_hor_gap = 10000;
							min_hor_gap_idx = -1;

							for (i = 0; i < numbers.size(); i++) {
								if (cur_idx == i || checked_panel_idx[i] > 0)
									continue;

								// horizontally close and vertically overlapped 
								/*	if( numbers[i*4] - numbers[cur_idx*4+2] < 100 &&
								 Math.min(numbers[i*4+3], numbers[cur_idx*4+3])-getMax(numbers[i*4+1], numbers[cur_idx*4+1]) > getMax(numbers[i*4+3]-numbers[i*4+1], numbers[cur_idx*4+3]-numbers[cur_idx*4+1])*0.8 ){

								 box_order_idx[idx++] = i;
								 cur_idx = i;
								 continue_while = 1;
								 checked_panel_idx[i] = 1;
								 break;
								 }
								 */
								if (Math.min(numbers.get(i).bottom, numbers.get(cur_idx).bottom) - Math.max(numbers.get(i).top, numbers.get(cur_idx).top) > Math.max(numbers.get(i).bottom - numbers.get(i).top, numbers.get(cur_idx).bottom - numbers.get(cur_idx).top) * 0.8) {

									if (numbers.get(i).left - numbers.get(cur_idx).right < min_hor_gap) {
										min_hor_gap = numbers.get(i).left - numbers.get(cur_idx).right; 
										min_hor_gap_idx = i;
									}
								}

							}

							if (min_hor_gap_idx >= 0) {
								box_order_idx[idx++] = min_hor_gap_idx;
								cur_idx = min_hor_gap_idx;
								continue_while = true;
								checked_panel_idx[min_hor_gap_idx] = 1;
								continue;
							}

							// new line or at the last box. 
							if (continue_while == false) {

								// check boxes with the left-most box in previous row.
								cur_idx = left_top_box_idx;

								j = 1000;
								first_col_box_idx = -1;
								for (i = 0; i < numbers.size(); i++) {
									if (cur_idx == i || checked_panel_idx[i] > 0)
										continue;

									// vertically close and horizontally overlapped 
									if (numbers.get(i).top - numbers.get(cur_idx).bottom < 100) { //&&
										//	Math.min(numbers[i*4+2], numbers[cur_idx*4+2])-getMax(numbers[i*4], numbers[cur_idx*4]) > getMax(numbers[i*4+2]-numbers[i*4], numbers[cur_idx*4+2]-numbers[cur_idx*4])*0.8 ){
										if (numbers.get(i).left < j) {
											j = numbers.get(i).left;
											first_col_box_idx = i;
										}
									}
								}

								// found a box. Otherwise, there's no more boxes. 
								if (first_col_box_idx != -1) {
									box_order_idx[idx++] = first_col_box_idx;
									cur_idx = first_col_box_idx;
									continue_while = true;
									checked_panel_idx[first_col_box_idx] = 1;
									left_top_box_idx = first_col_box_idx;
								}
							}
						}

						if( finalLabelSet.upper_or_lower_case == true )
							prev_box_label = 'a' - 1;
						else
							prev_box_label = 'A' - 1;
						
						for (i = 0; i < numbers.size(); i++) {
							// check out whether or not current box is labeled.
							// if so, store the label. 

							if (box_order_idx[i] < 0)
								continue;
								
							// box_order_idx has indices of ordered panels from left to right/top to bottom. So prev_box_label has index of the first unlabeled box.
							// this works if first box in the ordered idx has label but in case the first box has no label it causes problem. 
							// Suppose third box has label 'a', but since it has no label prev_box_label still is 'a'-1, and previous code just label the first box by 'a' that will be overwritten by already labeled third panel 'a'.
							// this cause less split boxes than image box detection result. 
							// fixed on Jan. 16, 2014.
							
							if (named_panel.get(box_order_idx[i]) > 0) {
								prev_box_label = named_panel_label.get(box_order_idx[i]);
								continue;
							}

							// prevent out of index error in accessing checked_label array.
							if( prev_box_label > 121 )
								prev_box_label = 121;
							
							if( prev_box_label > 89 && finalLabelSet.upper_or_lower_case == false )
								prev_box_label = 89;
														
							// find next unused labels.
							while ( (prev_box_label+1 >= 97 && checked_label[prev_box_label - 96] > 0) || (prev_box_label+1 < 97 && checked_label[prev_box_label - 64] > 0) ){
								prev_box_label++;
								
								if( prev_box_label > 121 || (prev_box_label < 97 && prev_box_label > 89) )
									break;
							}
							
							if( prev_box_label > 121 )
								prev_box_label = 121;
							
							if( prev_box_label > 89 && finalLabelSet.upper_or_lower_case == false )
								prev_box_label = 89;
							
							// if not labeled then label it with it's left neighbor's label. The next label of the neighbor's label.
							named_panel_label.set(box_order_idx[i], (char) (prev_box_label + 1));
							named_panel.set(box_order_idx[i], 1);
							if( prev_box_label+1 >= 97 )
								checked_label[prev_box_label-96] = 1;
							if( prev_box_label+1 < 97 )
								checked_label[prev_box_label-64] = 1;
							

							// if the assigned label is already used, then don't label current box.
							// e.g., panel A has two boxes and the algorithm detected two rectangle.
							// the second box has no label in it and no neighbor OCRed label.
							// first try to label it B.
							// But, in case B is already used to label the real panel B, then don't label current box.
							
						//	System.out.println("Prev: " + prev_box_label);
							
						//	if ( (prev_box_label+1 >= 97 && checked_label[prev_box_label - 96] > 0) || (prev_box_label+1 < 97 && checked_label[prev_box_label - 64] > 0) )
						//		named_panel.set(box_order_idx[i], 0);

							prev_box_label++;
							
						//	while( prev_box_label - 96 > 25 || prev_box_label - 64 > 25 )
						//		prev_box_label--;
							
							// to make sure that checked_label[prev_box_label -96 or 64] doesn't make any out of index error. Just label over 26 panels by Z or z (overwrite, maximum panels would be 26), according to upper_or_lower_case.
							if( prev_box_label > 121 )
								prev_box_label = 121;
							
							if( prev_box_label > 89 && finalLabelSet.upper_or_lower_case == false )
								prev_box_label = 89;
						}
					}

					for (i = 0; i < numbers.size(); i++) {
						if (split_box_idx.get(i) > 0 || named_panel.get(i) == 0)
							continue;

						if (numbers.get(i).left >= width || numbers.get(i).right >= width
								|| numbers.get(i).top >= height
								|| numbers.get(i).bottom >= height
								|| numbers.get(i).right <= numbers.get(i).left
								|| numbers.get(i).bottom <= numbers.get(i).top)
							continue;

						// split panels that have not been split from previous steps. 
						cvSetImageROI(inImg, cvRect(numbers.get(i).left,	numbers.get(i).top, Math.min(numbers.get(i).right - numbers.get(i).left + 1, inImg.width() - 1), Math.min(numbers.get(i).bottom	- numbers.get(i).top + 1, inImg.height() - 1)));
						
						IplImage cropped = cvCreateImage(cvSize(Math.min(numbers.get(i).right - numbers.get(i).left + 1, inImg.width() - 1), Math.min(numbers.get(i).bottom - numbers.get(i).top + 1, inImg.height() - 1)), inImg.depth(), inImg.nChannels());
						cvCopy(inImg, cropped, null);

						panel_name = "";
						panel_name = out_fname;

						panel_name = panel_name.split(".jpg")[0] + '_' + named_panel_label.get(i) + ".jpg";
						
						if( alphaStr.indexOf(named_panel_label.get(i)) != -1 ){
							cvSaveImage(panel_name, cropped);
							final_label_list += named_panel_label.get(i);
							num_split_panels++;
						//	printf("save 4: %c\n", named_panel_label[i]);
							
							Final_Panel tmp_panel = new Final_Panel();
							tmp_panel.left = numbers.get(i).left;
							tmp_panel.top = numbers.get(i).top;
							tmp_panel.right = numbers.get(i).right;
							tmp_panel.bottom = numbers.get(i).bottom;
							tmp_panel.label = named_panel_label.get(i);
							
							finalSplitPanels.add(tmp_panel);
						}

						cvReleaseImage(cropped); cropped = null;
						cvResetImageROI(inImg);

						split_box_idx.set(i, 1);						
					}
				}
			}
		}

		cvReleaseImage(img); img = null;
		cvReleaseImage(inImg); inImg = null;
		
		if (resized_img != null){
			cvReleaseImage(resized_img); resized_img = null;
		}

		String returnStr = num_split_panels + ":::";

		for(i=0; i<num_split_panels; i++){
			if( i < num_split_panels-1 )
				returnStr += final_label_list.charAt(i) + ",";
			else
				returnStr += final_label_list.charAt(i);
		}

	/*
		String ground_fname = "";
		ground_fname = imgName;
		
		ground_fname = ground_fname.replaceAll("test_set", "xml");
		ground_fname = ground_fname.replaceAll(".jpg", ".xml");
		
		utilFunc.evaluatePanelSplittingResult(finalSplitPanels, ground_fname, evalNumbers);
	*/
		
		
		// output creation for ImageCLEF2013 multipanel separation task
	/*	
		FileWriter writerCLEF = new FileWriter("F:/DYou/workspace/iMEDLINE/samples/splitResult.xml", true);
		writerCLEF.write("   <annotation>" + "\n");
		writerCLEF.write("      <filename>" + fileName + "</filename>\n");
		
		for(i=0; i<finalSplitPanels.size(); i++){
			writerCLEF.write("      <object>\n");
			writerCLEF.write("         <point x=\"" + finalSplitPanels.get(i).left + "\"" + " y=\"" + finalSplitPanels.get(i).top + "\" />\n");
			writerCLEF.write("         <point x=\"" + finalSplitPanels.get(i).right + "\"" + " y=\"" + finalSplitPanels.get(i).top + "\" />\n");
			writerCLEF.write("         <point x=\"" + finalSplitPanels.get(i).left + "\"" + " y=\"" + finalSplitPanels.get(i).bottom + "\" />\n");
			writerCLEF.write("         <point x=\"" + finalSplitPanels.get(i).right + "\"" + " y=\"" + finalSplitPanels.get(i).bottom + "\" />\n");
			writerCLEF.write("      </object>\n");
		}
		
		writerCLEF.write("   </annotation>" + "\n");		
		writerCLEF.close();
	*/	
		return returnStr;
	}

	// Added by Zhiyun
	public ArrayList<Final_Panel>  panelSplitter(String imgName,
			 Classifier OCR_model,
			 int no_text_panel, String text_labels, 
			 ArrayList<Integer> rectangles/*,			// for debug only		  
			 Core.PanelSplitter.evalCount evalNumbers*/) {

		int i, j, k, idx;
		int height, width;
		int setNo, match_label_no, unmatch_label_no;
		
		int [] label_order = new int [26];
		int [] ori_label_idx = new int [26];
		int [] sorted_label_idx = new int [26];		// initialize with -1
		String final_label_list = "";
		
		ArrayList<Final_Panel> finalSplitPanels = new ArrayList<Final_Panel>();
		
		
		// added on Feb. 19, 2014
		if( rectangles.size() >= 160 )
		return finalSplitPanels;
		
		
		ArrayList<UtilFunctions.BBoxCoor> numbers = new ArrayList<UtilFunctions.BBoxCoor>(); 
		
		for(i = 0; i<rectangles.size(); i+=4){
		UtilFunctions.BBoxCoor each_rect = new UtilFunctions.BBoxCoor();
		
		each_rect.left = rectangles.get(i);
		each_rect.top = rectangles.get(i+1);
		each_rect.right = rectangles.get(i+2);
		each_rect.bottom = rectangles.get(i+3);
		
		numbers.add(each_rect);		
		}
		
		
		IplImage inImg = cvLoadImage(imgName);
		
		//	cvSaveImage(out_fname, inImg);
		
		IplImage img, resized_img;
		ArrayList<PanelLabelDetection.PanelLabelInfo_Final> candLabelSet = new ArrayList<PanelLabelDetection.PanelLabelInfo_Final>();
		
		PanelLabelDetection.PanelLabelInfo_Final finalLabelSet = new PanelLabelDetection.PanelLabelInfo_Final();
		int num_split_panels = 0;
		
		height = inImg.height();
		width = inImg.width();
		
		// need to adjust the right and bottom coordinate.
		// the results include width and height as maximum coordinates, but functions allow only
		// up to width-1 and height-1 coordinates. 
		for (i = 0; i < numbers.size(); i++) {
		if (numbers.get(i).right >= width)
		numbers.get(i).right = width - 1;
		
		if (numbers.get(i).bottom >= height)
		numbers.get(i).bottom = height - 1;
		}
		
		// 200% enlargement for small image and color conversion
		resized_img = null;
		img = cvCreateImage(cvGetSize(inImg), IPL_DEPTH_8U, 1);
		
		if (inImg.nChannels() == 3 )
		cvCvtColor(inImg, img, CV_RGB2GRAY);			
		else
		cvCopy(inImg, img, null);
		
		//	Arrays.fill(candLabelSet, 0);
		
		PanelLabelDetection panelLabelDetectionRef = new PanelLabelDetection();
		
		
		// panel label detection. 
		// candLabelSet has all candidate label sets and the setNo denotes the total cand numbers. 
		if (width < 1000 && height < 1000) {
		resized_img = cvCreateImage(cvSize(width * 2, height * 2), IPL_DEPTH_8U, 1);
		cvResize(img, resized_img, 1);
		panelLabelDetectionRef.panelLabelDetection(resized_img, true, candLabelSet, OCR_model);			
		} else
		panelLabelDetectionRef.panelLabelDetection(img, false, candLabelSet, OCR_model);
		
		
		setNo = panelLabelDetectionRef.setNo;		
		
		
		{
		int max_match_set_idx = 0;
		int max_match_no, max_unmatch_no;
		
		int the_idx = 0;
		int pos_x, pos_y;
		int min_dist;
		int x_interval, y_interval, main_pos;
		
		int x_diff, y_diff;
		int left_margin, top_margin, right_margin, bottom_margin;
		
		int [] label_pos = new int [9];
		ArrayList<Integer> named_panel;
		ArrayList<Integer> split_box_idx;
		ArrayList<Integer> label_idx_in_one_box;
		
		int [] checked_label = new int [26];
		String panel_name = "";
		char the_label = 0;
		ArrayList<Character> named_panel_label = null;
		
		boolean new_panel_added, found_label;
		
		int no_already_split = 0;
		
		// indicate whether a panel in numbers array is labeled or not.  
		named_panel = new ArrayList<Integer>();
		split_box_idx = new ArrayList<Integer>();
		named_panel_label = new ArrayList<Character>();
		label_idx_in_one_box = new ArrayList<Integer>();
		
		for(i=0; i<numbers.size();i++){
		named_panel.add(0);
		split_box_idx.add(0);
		named_panel_label.add(' ');
		}
		
		// find a candidate that has the most identical labels with text labels.
		// just match text labels with OCR results. 
		
		max_match_no = max_unmatch_no = 0;
		///	max_case = 0;
		//	for(case_idx=0; case_idx < 1; case_idx++)
		if (no_text_panel > 0) {
		for (i = 0; i < setNo; i++) {
			match_label_no = unmatch_label_no = 0;
			Arrays.fill(checked_label, 0);
			for (j = 0; j < candLabelSet.get(i).noLabel; j++) {
				if (candLabelSet.get(i).labels.get(j).label == 0)
					continue;
										
				if( text_labels.indexOf(candLabelSet.get(i).labels.get(j).label) != -1 ){
				
					if (candLabelSet.get(i).labels.get(j).label >= 97) {
						if (checked_label[Math.min(candLabelSet.get(i).labels.get(j).label - 97, 25)] == 0) {
							match_label_no++;
							checked_label[Math.min(candLabelSet.get(i).labels.get(j).label - 97, 25)] = 1;
						}
					}
		
					if (candLabelSet.get(i).labels.get(j).label < 97 && candLabelSet.get(i).labels.get(j).label >= 65) {
						if (checked_label[Math.min(candLabelSet.get(i).labels.get(j).label - 65, 25)] == 0) {
							match_label_no++;
							checked_label[Math.min(candLabelSet.get(i).labels.get(j).label - 65, 25)] = 1;
						}
					}
				} else
					unmatch_label_no++;
			}
		
			if (match_label_no > max_match_no) {
				max_match_no = match_label_no;
				max_unmatch_no = unmatch_label_no;
				max_match_set_idx = i;
			}
		
			if (match_label_no == max_match_no && max_unmatch_no > unmatch_label_no) {
				max_unmatch_no = unmatch_label_no;
				max_match_set_idx = i;
			}
		}
		}
		
		Arrays.fill(checked_label, 0);
		
		// found a cand set that has the most identical labels with text labels. 
		if (max_match_no > 0) {
		
		// eliminate noise labels (unmatched labels) from the chosen candidate set. 
						
		if (candLabelSet.get(max_match_set_idx).noLabel > no_text_panel || candLabelSet.get(max_match_set_idx).noLabel > max_match_no) {
			for (j = 0; j < candLabelSet.get(max_match_set_idx).noLabel; j++) {
		
				if (candLabelSet.get(max_match_set_idx).labels.get(j).label == 0)
					continue;
				
				if ( text_labels.indexOf(candLabelSet.get(max_match_set_idx).labels.get(j).label) == -1 )
					candLabelSet.get(max_match_set_idx).labels.get(j).label = 0;
				else 
				{
					//	*ch_str = 95;
					for (i = 0; i < candLabelSet.get(max_match_set_idx).noLabel; i++) {
						if (i == j || candLabelSet.get(max_match_set_idx).labels.get(i).label == 0)
							continue;
		
						// in case the set has multiple identical labels, then pick a label with largest msg_1.
						if (candLabelSet.get(max_match_set_idx).labels.get(i).label == candLabelSet.get(max_match_set_idx).labels.get(j).label) {
							if (candLabelSet.get(max_match_set_idx).labels.get(i).msg_1 > candLabelSet.get(max_match_set_idx).labels.get(j).msg_1) {
								candLabelSet.get(max_match_set_idx).labels.get(j).label = 0;
								break;
							} else
								candLabelSet.get(max_match_set_idx).labels.get(i).label = 0;
						}
					}
				}
			}
		}
		
		finalLabelSet.upper_or_lower_case = candLabelSet.get(max_match_set_idx).upper_or_lower_case;
		finalLabelSet.labels = new ArrayList<PanelLabelDetection.PanelLabelInfo>();
		
		for (i = 0; i < candLabelSet.get(max_match_set_idx).noLabel; i++) {
			if (candLabelSet.get(max_match_set_idx).labels.get(i).label == 0)
				continue;
			
			PanelLabelDetection.PanelLabelInfo tmp_Info = new PanelLabelDetection.PanelLabelInfo();
			tmp_Info = candLabelSet.get(max_match_set_idx).labels.get(i);
			
			finalLabelSet.labels.add(tmp_Info);				
			
		//	System.arraycopy(candLabelSet.get(max_match_set_idx).labels, i, finalLabelSet.labels, finalLabelSet.noLabel, 1);
		//	finalLabelSet.noLabel++;
		}
		
		finalLabelSet.noLabel = finalLabelSet.labels.size();
		// count valid labels that can be used for labeling.
		
		for (j = 0; j < finalLabelSet.noLabel; j++) {
			if ( finalLabelSet.labels.get(j).label == 0 )
				continue;
		}
		
		
		// panel splitting with multiple panel labels in it.
		
		new_panel_added = true;
		
		// for left to right arrangement only.
		// need to implement top to bottom arrangement. 
		//	if( no_panel == 1 && no_valid_labels > 1 )
		{
			int init_num = numbers.size();
			
			while( new_panel_added == true ) {
		
				new_panel_added = false;
				for (i = 0; i < Math.min(init_num*2, numbers.size()); i++) {
					label_idx_in_one_box.clear();
					
					for (j = 0; j < finalLabelSet.noLabel; j++) {
						if (finalLabelSet.labels.get(j).label == 0)
							continue;
		
						if (finalLabelSet.labels.get(j).cen_x > numbers.get(i).left && finalLabelSet.labels.get(j).cen_y > numbers.get(i).top && finalLabelSet.labels.get(j).cen_x < numbers.get(i).right && finalLabelSet.labels.get(j).cen_y < numbers.get(i).bottom) {
							label_idx_in_one_box.add(j);									
						}
					}
		
					if (label_idx_in_one_box.size() > 2) {
		
						// sort the OCRed labels.
						idx = 0;
						Arrays.fill(label_order, 0);
						for (j = 0; j < label_idx_in_one_box.size(); j++) {
		
							label_order[idx] = finalLabelSet.labels.get(label_idx_in_one_box.get(j)).bottom;
							ori_label_idx[idx++] = j;
							
							if( idx > 25 )
								idx = 25;
						}
		
						utilFunc.sorting_min_2_max(label_order, sorted_label_idx, idx);
		
						left_margin = top_margin = right_margin = bottom_margin = 1000;
						for (j = 0; j < label_idx_in_one_box.size(); j++) {
							left_margin = Math.min(finalLabelSet.labels.get(label_idx_in_one_box.get(j)).left - numbers.get(i).left, left_margin);
							top_margin = Math.min(finalLabelSet.labels.get(label_idx_in_one_box.get(j)).top - numbers.get(i).top, top_margin);
							right_margin = Math.min(numbers.get(i).right - finalLabelSet.labels.get(label_idx_in_one_box.get(j)).right, right_margin);
							bottom_margin = Math.min(numbers.get(i).bottom - finalLabelSet.labels.get(label_idx_in_one_box.get(j)).bottom, bottom_margin);
						}
		
						// bottom_left (right) label pattern. 
						if (bottom_margin < top_margin && bottom_margin < 100 && ((left_margin < right_margin && left_margin < 100) || (right_margin < left_margin && right_margin < 100))) {
		
							for (j = label_idx_in_one_box.size() - 1; j > 0; j--) {
								if (Math.abs(finalLabelSet.labels.get(label_idx_in_one_box.get(ori_label_idx[sorted_label_idx[j]])).cen_y - finalLabelSet.labels.get(label_idx_in_one_box.get(ori_label_idx[sorted_label_idx[j - 1]])).cen_y) > 100) {
									
									UtilFunctions.BBoxCoor tmp_coor = new UtilFunctions.BBoxCoor();
									
									tmp_coor.left = numbers.get(i).left;
									tmp_coor.top = finalLabelSet.labels.get(label_idx_in_one_box.get(ori_label_idx[sorted_label_idx[j - 1]])).bottom + bottom_margin;
									tmp_coor.right = numbers.get(i).right;
									tmp_coor.bottom = numbers.get(i).bottom; 
									
									numbers.get(i).bottom = tmp_coor.top - 1;
									
									numbers.add(tmp_coor);
									named_panel.add(0);
									split_box_idx.add(0);
									named_panel_label.add(' ');
									new_panel_added = true;
								}
							}
						}
		
						// top_left (right) label pattern. 
						if (top_margin < bottom_margin && top_margin < 100
								&& ((left_margin < right_margin
										&& left_margin < 100)
										|| (right_margin < left_margin
												&& right_margin < 100))) {
		
							for (j = label_idx_in_one_box.size() - 1; j > 0; j--) {
								if (Math.abs(finalLabelSet.labels.get(label_idx_in_one_box.get(ori_label_idx[sorted_label_idx[j]])).cen_y - finalLabelSet.labels.get(label_idx_in_one_box.get(ori_label_idx[sorted_label_idx[j - 1]])).cen_y) > 100) {
									UtilFunctions.BBoxCoor tmp_coor = new UtilFunctions.BBoxCoor();
									
									tmp_coor.left = numbers.get(i).left;
									tmp_coor.top = finalLabelSet.labels.get(label_idx_in_one_box.get(ori_label_idx[sorted_label_idx[j]])).top - top_margin;
									tmp_coor.right = numbers.get(i).right; 
									tmp_coor.bottom = numbers.get(i).bottom;
									
									numbers.get(i).bottom = tmp_coor.top - 1;
									
									numbers.add(tmp_coor);
									named_panel.add(0);
									split_box_idx.add(0);
									named_panel_label.add(' ');
									
									new_panel_added = true;											
								}
							}
						}
					}
				}					
			}
		}
		//	else
		
		{
			new_panel_added = true;
			int init_num = numbers.size();
			
			while (new_panel_added == true ) {
		
				new_panel_added = false;
		
				for (i = 0; i < Math.min(init_num*2, numbers.size()); i++) {
					
					label_idx_in_one_box.clear();
		
					for (j = 0; j < finalLabelSet.noLabel; j++) {
						if (finalLabelSet.labels.get(j).label == 0)
							continue;
		
						if (finalLabelSet.labels.get(j).cen_x > numbers.get(i).left && finalLabelSet.labels.get(j).cen_y > numbers.get(i).top && finalLabelSet.labels.get(j).cen_x < numbers.get(i).right && finalLabelSet.labels.get(j).cen_y < numbers.get(i).bottom) {
							label_idx_in_one_box.add(j);
						}
					}
		
					// two labels in one box
					if (label_idx_in_one_box.size() >= 2) {
						// find the arrangement of the labels. compare the x and y coordinates of the center.
						// if x_diff is very small, then vertical arrangement. 
						// if y_diff is very small, then horizontal arrangement. 
		
						// find where the labels are located. Usually on a corner of the rectangle. 
						// if conditions met, then split it into two and store the new rectangle and modify the original one. 
						x_diff = Math.abs(finalLabelSet.labels.get(label_idx_in_one_box.get(0)).cen_x - finalLabelSet.labels.get(label_idx_in_one_box.get(1)).cen_x);
						y_diff = Math.abs(finalLabelSet.labels.get(label_idx_in_one_box.get(0)).cen_y - finalLabelSet.labels.get(label_idx_in_one_box.get(1)).cen_y);
		
						// left to right arrangement
						if (y_diff < finalLabelSet.labels.get(label_idx_in_one_box.get(0)).bottom - finalLabelSet.labels.get(label_idx_in_one_box.get(0)).top) {
		
							idx = 0;
							Arrays.fill(label_order, 0);
							Arrays.fill(ori_label_idx, 0);
							Arrays.fill(sorted_label_idx, 0);
							
							for (j = 0; j < label_idx_in_one_box.size(); j++) {
								label_order[idx] = finalLabelSet.labels.get(label_idx_in_one_box.get(j)).right;
								ori_label_idx[idx++] = j;
								
								if( idx > 25 )
									idx = 25;
							}
		
							utilFunc.sorting_max_2_min_INT(label_order, sorted_label_idx, idx);
		
							left_margin = right_margin = 1000;
							for (j = 0; j < label_idx_in_one_box.size(); j++) {
								left_margin = Math.min(left_margin, finalLabelSet.labels.get(label_idx_in_one_box.get(j)).left - numbers.get(i).left);
								right_margin = Math.min(right_margin, numbers.get(i).right - finalLabelSet.labels.get(label_idx_in_one_box.get(j)).right);
							}
		
							// left corner location
							if (left_margin < right_margin && left_margin < 100) {
								for (j = 0; j < label_idx_in_one_box.size() - 1; j++) {
									UtilFunctions.BBoxCoor tmp_coor = new UtilFunctions.BBoxCoor();
									
									tmp_coor.left = finalLabelSet.labels.get(label_idx_in_one_box.get(ori_label_idx[sorted_label_idx[j]])).left - left_margin;
									tmp_coor.top = numbers.get(i).top;
									tmp_coor.right = numbers.get(i).right;
									tmp_coor.bottom = numbers.get(i).bottom;
									
									numbers.get(i).right = tmp_coor.left - 1;
									
									numbers.add(tmp_coor);
									named_panel.add(0);
									split_box_idx.add(0);
									named_panel_label.add(' ');
									
								}
							} else
							// right corner location
							if (right_margin < left_margin && right_margin < 100) {
		
								for (j = 1; j < label_idx_in_one_box.size(); j++) {
									UtilFunctions.BBoxCoor tmp_coor = new UtilFunctions.BBoxCoor();
									tmp_coor.left = finalLabelSet.labels.get(label_idx_in_one_box.get(ori_label_idx[sorted_label_idx[j]])).right + right_margin;
									tmp_coor.top = numbers.get(i).top;
									tmp_coor.right = numbers.get(i).right;
									tmp_coor.bottom = numbers.get(i).bottom;
									
									numbers.get(i).right = tmp_coor.left - 1;
									
									numbers.add(tmp_coor);
									named_panel.add(0);
									split_box_idx.add(0);
									named_panel_label.add(' ');
								}
							}
						}
		
						// top to bottom arrangement
						if (x_diff < finalLabelSet.labels.get(label_idx_in_one_box.get(0)).right - finalLabelSet.labels.get(label_idx_in_one_box.get(0)).left) {
		
							idx = 0;
							Arrays.fill(label_order, 0);
							Arrays.fill(ori_label_idx, 0);
							Arrays.fill(sorted_label_idx, 0);
							
							for (j = 0; j < label_idx_in_one_box.size(); j++) {
		
								label_order[idx] = finalLabelSet.labels.get(label_idx_in_one_box.get(j)).bottom;
								ori_label_idx[idx++] = j;
								
								if( idx > 25 )
									idx = 25;
							}
		
							utilFunc.sorting_max_2_min_INT(label_order, sorted_label_idx, idx);
		
							top_margin = bottom_margin = 1000;
							for (j = 0; j < label_idx_in_one_box.size(); j++) {
								top_margin = Math.min(top_margin, finalLabelSet.labels.get(label_idx_in_one_box.get(j)).top - numbers.get(i).top);
								bottom_margin = Math.min(bottom_margin, numbers.get(i).bottom - finalLabelSet.labels.get(label_idx_in_one_box.get(j)).bottom);
							}
		
							// upper corner location.
							if (top_margin < bottom_margin
									&& top_margin < 100) {
		
								for (j = 0; j < label_idx_in_one_box.size() - 1;	j++) {
									UtilFunctions.BBoxCoor tmp_coor = new UtilFunctions.BBoxCoor();
									
									tmp_coor.left = numbers.get(i).left;
									tmp_coor.top = finalLabelSet.labels.get(label_idx_in_one_box.get(ori_label_idx[sorted_label_idx[j]])).top - top_margin;
									tmp_coor.right = numbers.get(i).right;
									tmp_coor.bottom = numbers.get(i).bottom;
									
									numbers.get(i).bottom = tmp_coor.top - 1;
									
									numbers.add(tmp_coor);		
									named_panel.add(0);
									split_box_idx.add(0);
									named_panel_label.add(' ');
								}
							}
		
							// lower corner location
							if (bottom_margin < top_margin && bottom_margin < 100) {
		
								for (j = 1; j < label_idx_in_one_box.size(); j++) {
									
									UtilFunctions.BBoxCoor tmp_coor = new UtilFunctions.BBoxCoor();
									tmp_coor.left = numbers.get(i).left;
									tmp_coor.top = finalLabelSet.labels.get(label_idx_in_one_box.get(ori_label_idx[sorted_label_idx[j]])).bottom + bottom_margin;
									tmp_coor.right = numbers.get(i).right;
									tmp_coor.bottom = numbers.get(i).bottom;
									
									numbers.get(i).bottom = tmp_coor.top-1;
									
									numbers.add(tmp_coor);
									named_panel.add(0);
									split_box_idx.add(0);
									named_panel_label.add(' ');
								}
							}
						}
					}
				}					
			}
		}
		
		// try to label each panel that contains only one label in it. 
		Arrays.fill(label_pos, 0);
		
						
		for (i = 0; i < numbers.size(); i++) {
			int no_labels_in_a_box = 0;
		
			for (j = 0; j < finalLabelSet.noLabel; j++) {
				if (finalLabelSet.labels.get(j).label == 0)
					continue;
		
				if (finalLabelSet.labels.get(j).cen_x > numbers.get(i).left && finalLabelSet.labels.get(j).cen_y > numbers.get(i).top && finalLabelSet.labels.get(j).cen_x < numbers.get(i).right && finalLabelSet.labels.get(j).cen_y < numbers.get(i).bottom) {
		
					the_label = finalLabelSet.labels.get(j).label;
					the_idx = j;
					no_labels_in_a_box++;
				}
			}
		
			// there's a only one label in the box
			// split and label it. 
			if (no_labels_in_a_box == 1) {
		
				if (numbers.get(i).left >= width || numbers.get(i).right >= width
						|| numbers.get(i).top >= height
						|| numbers.get(i).bottom >= height
						|| numbers.get(i).right <= numbers.get(i).left
						|| numbers.get(i).bottom <= numbers.get(i).top)
					continue;
		
				if (split_box_idx.get(i) > 0)
					continue;
		
				cvSetImageROI(inImg, cvRect(numbers.get(i).left,numbers.get(i).top, Math.min(numbers.get(i).right - numbers.get(i).left + 1, inImg.width() - 1), Math.min(numbers.get(i).bottom - numbers.get(i).top + 1, inImg.height() - 1)));
				IplImage cropped = cvCreateImage(cvSize(Math.min(numbers.get(i).right - numbers.get(i).left + 1, inImg.width() - 1), Math.min(numbers.get(i).bottom - numbers.get(i).top + 1, inImg.height() - 1)), inImg.depth(), inImg.nChannels());
				cvCopy(inImg, cropped, null);
		
				// if statement added on 12/19/2012
				if( alphaStr.indexOf(the_label) != -1 ){
					cvSaveImage(panel_name, cropped);
					final_label_list += the_label;
					num_split_panels++;
				//	printf("save 1: %c\n", the_label);
					
					Final_Panel tmp_panel = new Final_Panel();
					tmp_panel.left = numbers.get(i).left;
					tmp_panel.top = numbers.get(i).top;
					tmp_panel.right = numbers.get(i).right;
					tmp_panel.bottom = numbers.get(i).bottom;
					tmp_panel.label = the_label;
					
					finalSplitPanels.add(tmp_panel);
				}
		
				cvReleaseImage(cropped); cropped = null;
				cvResetImageROI(inImg);
		
				finalLabelSet.labels.get(the_idx).used = true;
				named_panel.set(i, 1);
				named_panel_label.set(i, finalLabelSet.labels.get(the_idx).label);
				split_box_idx.set(i, 1);
				no_already_split++;
		
				if( named_panel_label.get(i) >= 97 )
					checked_label[Math.min(named_panel_label.get(i) - 97, 25)] = 1;
				else
					if( named_panel_label.get(i) < 97 && named_panel_label.get(i) >= 65 )
						checked_label[Math.min(named_panel_label.get(i) - 65, 25)] = 1;
				
				
				// find the pattern of location of labels. 3*3 position mask. 
				if (finalLabelSet.labels.get(the_idx).cen_x < (numbers.get(i).right - numbers.get(i).left) / 3 + numbers.get(i).left)
					pos_x = 0;
				else if (finalLabelSet.labels.get(the_idx).cen_x > numbers.get(i).right - (numbers.get(i).right - numbers.get(i).left) / 3)
					pos_x = 2;
				else
					pos_x = 1;
		
				if (finalLabelSet.labels.get(the_idx).cen_y < (numbers.get(i).bottom - numbers.get(i).top) / 3 + numbers.get(i).top)
					pos_y = 0;
				else if (finalLabelSet.labels.get(the_idx).cen_y > numbers.get(i).bottom - (numbers.get(i).bottom - numbers.get(i).top) / 3)
					pos_y = 2;
				else
					pos_y = 1;
		
				label_pos[pos_y * 3 + pos_x]++;
			}
		}
		
		k = 0;
		main_pos = -1;
		for (i = 0; i < 9; i++) {
			if (label_pos[i] > 0) {
				main_pos = i;
				k++;
			}
		}
		
		// for now, only think about one label location pattern, e.g., all labels are located on the upper-left corner.
		// otherwise, don't use this information. 
		if (k > 1)
			main_pos = -1;
		
		// process if only one label position pattern detected. 
		// try to find labels out of the panels. 
		// assume that we just move the bounding box a little bit to contain a certain label near the box and satisfy
		// the pattern. The label is right label for the panel. 
		if (main_pos != -1) {
			for (i = 0; i < numbers.size(); i++) {
				if (named_panel.get(i) > 0)
					continue;
		
				found_label = false;
				min_dist = 10000;
		
				x_interval = (numbers.get(i).right - numbers.get(i).left) / 3;
				y_interval = (numbers.get(i).bottom - numbers.get(i).top) / 3;
		
				for (j = 0; j < finalLabelSet.noLabel; j++) {
		
					if (finalLabelSet.labels.get(j).used == true || finalLabelSet.labels.get(j).label == 0 )
						continue;
		
					switch (main_pos) {
					// upper left corner. the top or left can be moved to contain the label.
					// find the closest one.
					case 0:
						if (finalLabelSet.labels.get(j).cen_y > numbers.get(i).top && finalLabelSet.labels.get(j).cen_y < numbers.get(i).top + y_interval) {
		
							if (finalLabelSet.labels.get(j).cen_x < numbers.get(i).left && numbers.get(i).left - finalLabelSet.labels.get(j).cen_x < min_dist) {
								min_dist = numbers.get(i).left - finalLabelSet.labels.get(j).cen_x;
								the_idx = j;
								the_label = finalLabelSet.labels.get(j).label;
		
								found_label = true;
							}
						}
		
						if (finalLabelSet.labels.get(j).cen_x > numbers.get(i).left && finalLabelSet.labels.get(j).cen_x < numbers.get(i).left + x_interval) {
		
							if (finalLabelSet.labels.get(j).cen_y < numbers.get(i).top && numbers.get(i).top - finalLabelSet.labels.get(j).cen_y < min_dist) {
								min_dist = numbers.get(i).top - finalLabelSet.labels.get(j).cen_y;
								the_idx = j;
								the_label = finalLabelSet.labels.get(j).label;
		
								found_label = true;
							}
						}
		
						break;
		
						// lower-left corner.
					case 6:
						if (finalLabelSet.labels.get(j).cen_y < numbers.get(i).bottom && finalLabelSet.labels.get(j).cen_y > numbers.get(i).bottom - y_interval) {
		
							if (finalLabelSet.labels.get(j).cen_x < numbers.get(i).left && numbers.get(i).left - finalLabelSet.labels.get(j).cen_x < min_dist) {
								min_dist = numbers.get(i).left - finalLabelSet.labels.get(j).cen_x;
								the_idx = j;
								the_label = finalLabelSet.labels.get(j).label;
		
								found_label = true;
							}
						}
		
						if (finalLabelSet.labels.get(j).cen_x > numbers.get(i).left && finalLabelSet.labels.get(j).cen_x < numbers.get(i).left + x_interval) {
		
							if (finalLabelSet.labels.get(j).cen_y > numbers.get(i).bottom && finalLabelSet.labels.get(j).cen_y - numbers.get(i).bottom < min_dist) {
								min_dist = finalLabelSet.labels.get(j).cen_y - numbers.get(i).bottom;
								the_idx = j;
								the_label = finalLabelSet.labels.get(j).label;
		
								found_label = true;
							}
						}
		
						break;
		
					default:
						break;
					}
				}
		
				// if a label is found for this rectangle, then name the panel. 
				if (found_label == true ) {
		
					if (numbers.get(i).left >= width
							|| numbers.get(i).right >= width
							|| numbers.get(i).top >= height
							|| numbers.get(i).bottom >= height
							|| numbers.get(i).right <= numbers.get(i).left
							|| numbers.get(i).bottom <= numbers.get(i).top)
						continue;
		
					if (split_box_idx.get(i) > 0)
						continue;
		
					cvSetImageROI(inImg, cvRect(numbers.get(i).left,numbers.get(i).top,Math.min(numbers.get(i).right- numbers.get(i).left + 1,inImg.width() - 1),Math.min(numbers.get(i).bottom- numbers.get(i).top + 1,inImg.height() - 1)));
					IplImage cropped = cvCreateImage(cvSize(Math.min(numbers.get(i).right - numbers.get(i).left + 1,inImg.width() - 1),Math.min(numbers.get(i).bottom - numbers.get(i).top + 1, inImg.height() - 1)),inImg.depth(), inImg.nChannels());
					cvCopy(inImg, cropped, null);
		
					
					if( alphaStr.indexOf(the_label) != -1 ){
						final_label_list += the_label;
						num_split_panels++;
					//	printf("save 2: %c\n", the_label);
						
						Final_Panel tmp_panel = new Final_Panel();
						tmp_panel.left = numbers.get(i).left;
						tmp_panel.top = numbers.get(i).top;
						tmp_panel.right = numbers.get(i).right;
						tmp_panel.bottom = numbers.get(i).bottom;
						tmp_panel.label = the_label;
						
						finalSplitPanels.add(tmp_panel);
					}
		
		
					cvReleaseImage(cropped); cropped = null;
					cvResetImageROI(inImg);
		
					finalLabelSet.labels.get(the_idx).used = true;
					named_panel.set(i, 1);
					named_panel_label.set(i, finalLabelSet.labels.get(the_idx).label);
					split_box_idx.set(i, 1);
					no_already_split++;
		
					if( named_panel_label.get(i) >= 97 )
						checked_label[Math.min(named_panel_label.get(i)-97, 25)] = 1;
					else
						if( named_panel_label.get(i) < 97 && named_panel_label.get(i) >= 65 )
							checked_label[Math.min(named_panel_label.get(i)-65, 25)] = 1;
				}
			}
		}
		}
		
		
		// updated...
		// no panels are labeled.
		// panel labels are outside of panels. 
		// cause problems with awp328f7
		// applied only no panels are labeled before since all labels are outside of panels. 
		
		if (no_already_split == 0) {
		int type_idx;
		char [][] named_panel_label_tmp = new char [6][50];
		int [][] named_panel_label_tmp_idx = new int [6][50];
		int [] assigned_label_no = new int [6];
		
		//	Arrays.fill(named_panel_label_tmp, 0);
		//	Arrays.fill(named_panel_label_tmp_idx, 0);
		//	Arrays.fill(assigned_label_no, 0);
		
		for (type_idx = 0; type_idx < 6; type_idx++) {
		
			switch (type_idx) {
		
			case 0:
				for (i = 0; i < Math.min(numbers.size(), 50); i++) {
		
					if (named_panel.get(i) > 0)
						continue;
		
					for (j = 0; j < finalLabelSet.noLabel; j++) {
						if (finalLabelSet.labels.get(j).label == 0 || finalLabelSet.labels.get(j).used == true )
							continue;
		
						if (finalLabelSet.labels.get(j).cen_x > numbers.get(i).left && finalLabelSet.labels.get(j).cen_y < numbers.get(i).top && finalLabelSet.labels.get(j).cen_x < numbers.get(i).left + (numbers.get(i).right - numbers.get(i).left) / 3 && numbers.get(i).top - finalLabelSet.labels.get(j).cen_y < 100) {
							named_panel_label_tmp[type_idx][i] = finalLabelSet.labels.get(j).label;
							named_panel_label_tmp_idx[type_idx][i] = j;
							assigned_label_no[type_idx]++;
							break;
						}
					}
				}
		
				break;
		
			case 1:
		
				for (i = 0; i < Math.min(numbers.size(), 50); i++) {
		
					if (named_panel.get(i) > 0)
						continue;
		
					for (j = 0; j < finalLabelSet.noLabel; j++) {
						if (finalLabelSet.labels.get(j).label == 0 || finalLabelSet.labels.get(j).used == true )
							continue;
		
						if (finalLabelSet.labels.get(j).cen_x < numbers.get(i).left && finalLabelSet.labels.get(j).cen_y > numbers.get(i).top && finalLabelSet.labels.get(j).cen_y < numbers.get(i).top + (numbers.get(i).bottom - numbers.get(i).top) / 3 && numbers.get(i).left - finalLabelSet.labels.get(j).cen_x < 100) {
							named_panel_label_tmp[type_idx][i] = finalLabelSet.labels.get(j).label;
							named_panel_label_tmp_idx[type_idx][i] = j;
							assigned_label_no[type_idx]++;
							break;
						}
					}
				}
		
				break;
		
			case 2:
		
				for (i = 0; i < Math.min(numbers.size(), 50); i++) {
		
					if (named_panel.get(i) > 0)
						continue;
		
					for (j = 0; j < finalLabelSet.noLabel; j++) {
						if (finalLabelSet.labels.get(j).label == 0 || finalLabelSet.labels.get(j).used == true)
							continue;
		
						if (finalLabelSet.labels.get(j).cen_x < numbers.get(i).left && finalLabelSet.labels.get(j).cen_y < numbers.get(i).bottom && finalLabelSet.labels.get(j).cen_y > numbers.get(i).bottom - (numbers.get(i).bottom - numbers.get(i).top) / 3 && numbers.get(i).left - finalLabelSet.labels.get(j).cen_x < 100) {
							named_panel_label_tmp[type_idx][i] = finalLabelSet.labels.get(j).label;
							named_panel_label_tmp_idx[type_idx][i] = j;
							assigned_label_no[type_idx]++;
							break;
						}
					}
				}
		
				break;
		
			case 3:
		
				for (i = 0; i < Math.min(numbers.size(),50); i++) {
		
					if (named_panel.get(i) > 0)
						continue;
		
					for (j = 0; j < finalLabelSet.noLabel; j++) {
						if (finalLabelSet.labels.get(j).label == 0 || finalLabelSet.labels.get(j).used == true )
							continue;
		
						if (finalLabelSet.labels.get(j).cen_x > numbers.get(i).left && finalLabelSet.labels.get(j).cen_y > numbers.get(i).bottom && finalLabelSet.labels.get(j).cen_x < numbers.get(i).left + (numbers.get(i).right - numbers.get(i).left) / 3 && finalLabelSet.labels.get(j).cen_y - numbers.get(i).bottom < 100) {
							named_panel_label_tmp[type_idx][i] = finalLabelSet.labels.get(j).label;
							named_panel_label_tmp_idx[type_idx][i] = j;
							assigned_label_no[type_idx]++;
							break;
						}
					}
				}
		
				break;
		
			case 4:
		
				for (i = 0; i < Math.min(numbers.size(), 50); i++) {
		
					if (named_panel.get(i) > 0)
						continue;
		
					for (j = 0; j < finalLabelSet.noLabel; j++) {
						if (finalLabelSet.labels.get(j).label == 0 || finalLabelSet.labels.get(j).used == true )
							continue;
		
						if (finalLabelSet.labels.get(j).cen_x < numbers.get(i).left && finalLabelSet.labels.get(j).cen_x > numbers.get(i).left - 50 && finalLabelSet.labels.get(j).cen_y > numbers.get(i).top - 50 && finalLabelSet.labels.get(j).cen_y < numbers.get(i).top + 50) {
		
							named_panel_label_tmp[type_idx][i] = finalLabelSet.labels.get(j).label;
							named_panel_label_tmp_idx[type_idx][i] = j;
							assigned_label_no[type_idx]++;
							break;
						}
					}
				}
		
				break;
		
			case 5: // out, bottom, center of the panel.
		
				for (i = 0; i < Math.min(numbers.size(),50); i++) {
		
					if (named_panel.get(i) > 0)
						continue;
		
					for (j = 0; j < finalLabelSet.noLabel; j++) {
						if (finalLabelSet.labels.get(j).label == 0 || finalLabelSet.labels.get(j).used == true )
							continue;
		
						if (finalLabelSet.labels.get(j).cen_x > numbers.get(i).left + (numbers.get(i).right - numbers.get(i).left) / 3 && finalLabelSet.labels.get(j).cen_x < numbers.get(i).right - (numbers.get(i).right - numbers.get(i).left) / 3 && finalLabelSet.labels.get(j).cen_y - numbers.get(i).bottom < 100 && finalLabelSet.labels.get(j).cen_y - numbers.get(i).bottom > 0) {
		
							named_panel_label_tmp[type_idx][i] = finalLabelSet.labels.get(j).label;
							named_panel_label_tmp_idx[type_idx][i] = j;
							assigned_label_no[type_idx]++;
							break;
						}
					}
				}
		
				break;
		
			}
		}
		
		i = 0;
		j = 0;
		
		for (type_idx = 0; type_idx < 6; type_idx++) {
			if (assigned_label_no[type_idx] > i) {
				i = assigned_label_no[type_idx];
				j = type_idx;
			}
		}
		
		if (i > 0) {
			type_idx = j;
		
			for (i = 0; i < Math.min(numbers.size(),50); i++) {
				if (named_panel.get(i) > 0
						|| named_panel_label_tmp[type_idx][i] == 0
						|| split_box_idx.get(i) > 0)
					continue;
		
				if (numbers.get(i).left >= width || numbers.get(i).right >= width
						|| numbers.get(i).top >= height
						|| numbers.get(i).bottom >= height
						|| numbers.get(i).right <= numbers.get(i).left
						|| numbers.get(i).bottom <= numbers.get(i).top)
					continue;
		
				cvSetImageROI(inImg, cvRect(numbers.get(i).left,numbers.get(i).top,Math.min(numbers.get(i).right - numbers.get(i).left+ 1, inImg.width() - 1),Math.min(numbers.get(i).bottom- numbers.get(i).top + 1,inImg.height() - 1)));
				
				IplImage cropped = cvCreateImage(cvSize(Math.min(numbers.get(i).right - numbers.get(i).left+ 1, inImg.width() - 1),Math.min(numbers.get(i).bottom- numbers.get(i).top + 1,inImg.height() - 1)), inImg.depth(),inImg.nChannels());
				cvCopy(inImg, cropped, null);
		
				
				if( alphaStr.indexOf(named_panel_label_tmp[type_idx][i]) != -1 ){
					final_label_list += named_panel_label_tmp[type_idx][i];
					num_split_panels++;
				//	printf("save 3: %c\n", named_panel_label_tmp[type_idx][i]);
					
					Final_Panel tmp_panel = new Final_Panel();
					tmp_panel.left = numbers.get(i).left;
					tmp_panel.top = numbers.get(i).top;
					tmp_panel.right = numbers.get(i).right;
					tmp_panel.bottom = numbers.get(i).bottom;
					tmp_panel.label = named_panel_label_tmp[type_idx][i];
					
					finalSplitPanels.add(tmp_panel);							
				}
		
				cvReleaseImage(cropped); cropped = null;
				cvResetImageROI(inImg);
				
				finalLabelSet.labels.get(named_panel_label_tmp_idx[type_idx][i]).used = true;
				named_panel.set(i, 1);
				named_panel_label.set(i, named_panel_label_tmp[type_idx][i]);
				split_box_idx.set(i, 1);
				no_already_split++;
		
				if( named_panel_label.get(i) >= 97 )
					checked_label[Math.min(named_panel_label.get(i)-97, 25)] = 1;
				else
					if( named_panel_label.get(i) < 97 && named_panel_label.get(i) >= 65 )
						checked_label[Math.min(named_panel_label.get(i)-65, 25)] = 1;
				
			}
		}
		
		}
		
		
		// so far two cases are processed. 
		// 1. panels containing single label.
		// 2. panels that can contain a label if the box is expanded a little bit. 
		// 3. all labels are outside of panels. 
		
		// Try to assign a label based on the panels with labels and the position infomation of all the detected labels and panels.
		
		{
		int tmp_dist, min_box_dist;
		int cur_idx = 0;
		int left_top_box_idx = 0, first_col_box_idx = 0;
		int [] box_order_idx = new int [numbers.size()];
		int [] checked_panel_idx = new int [numbers.size()];
		
		char first_named_label;
		
		char prev_box_label;
		
		boolean left_to_right_pattern, top_to_bottom_pattern;
		boolean continue_while;
		
		int min_hor_gap, min_hor_gap_idx;
		
		// store the indices of rectangles by left to right and top to bottom orders.
		
		Arrays.fill(box_order_idx, -1);
		Arrays.fill(checked_panel_idx, 0);
		
		left_to_right_pattern = top_to_bottom_pattern = false;
		// some rectangles don't have labels.
		// need to assign one
		
		j = 0;
		for (i = 0; i < numbers.size(); i++) {
			if (named_panel.get(i) > 0)
				j++;
		}
		
		//	if( no_valid_labels < no_panel ){
		
		// some panels don't have labels. 
		if (j < numbers.size() ) {
		
			// find the left-top box
		
			// find the top-left most box. 
			min_box_dist = 1000;
			for (i = 0; i < numbers.size(); i++) {
				tmp_dist = (int) Math.sqrt( (double) (numbers.get(i).left * numbers.get(i).left + numbers.get(i).top * numbers.get(i).top));
		
				if (tmp_dist < min_box_dist) {
					left_top_box_idx = i;
					min_box_dist = tmp_dist;
				}
			}
		
			// sort the OCRed labels.
			idx = 0;
			Arrays.fill(label_order, 0);
			Arrays.fill(ori_label_idx, 0);
		
			for (i = 0; i < finalLabelSet.noLabel; i++) {
				if (finalLabelSet.labels.get(i).label == 0)
					continue;
		
				label_order[idx] = finalLabelSet.labels.get(i).label;
				ori_label_idx[idx++] = i;
				
				if( idx > 25 )
					idx = 25;
			}
		
			utilFunc.sorting_min_2_max(label_order, sorted_label_idx, idx);
		
			// find the arrangement pattern. 
			for (i = 0; i < idx - 1; i++) {
				if (finalLabelSet.labels.get(ori_label_idx[sorted_label_idx[i + 1]]).label - finalLabelSet.labels.get(ori_label_idx[sorted_label_idx[i]]).label != 1)
					continue;
		
				x_diff = Math.abs(finalLabelSet.labels.get(ori_label_idx[sorted_label_idx[i]]).cen_x - finalLabelSet.labels.get(ori_label_idx[sorted_label_idx[i + 1]]).cen_x);
				y_diff = Math.abs(finalLabelSet.labels.get(ori_label_idx[sorted_label_idx[i]]).cen_y - finalLabelSet.labels.get(ori_label_idx[sorted_label_idx[i + 1]]).cen_y);
		
				if (y_diff < finalLabelSet.labels.get(ori_label_idx[sorted_label_idx[i]]).bottom - finalLabelSet.labels.get(ori_label_idx[sorted_label_idx[i]]).top) 
					left_to_right_pattern = true;
		
				if (x_diff < finalLabelSet.labels.get(ori_label_idx[sorted_label_idx[i]]).right - finalLabelSet.labels.get(ori_label_idx[sorted_label_idx[i]]).left)
					top_to_bottom_pattern = true;
		
				if (left_to_right_pattern == true || top_to_bottom_pattern == true )
					break;
			}
		
			//	if( left_to_right_pattern )
			{
		
				// find the first OCRed label. Most probably A or B (C, etc...)
				first_named_label = 127;
				for (i = 0; i < numbers.size(); i++) {
					if (named_panel.get(i) > 0 && named_panel_label.get(i) < first_named_label) {
						first_named_label = named_panel_label.get(i);
					}
				}
		
				cur_idx = left_top_box_idx;
				continue_while = true;
				checked_panel_idx[cur_idx] = 1;
				box_order_idx[0] = left_top_box_idx;
				idx = 1;
		
				// try to find rectangle indices closet to current box. 
				// find the order of the boxes from left to right, top to bottom.
				while (continue_while == true ) {
		
					continue_while = false;
					min_hor_gap = 10000;
					min_hor_gap_idx = -1;
		
					for (i = 0; i < numbers.size(); i++) {
						if (cur_idx == i || checked_panel_idx[i] > 0)
							continue;
		
						// horizontally close and vertically overlapped 
						/*	if( numbers[i*4] - numbers[cur_idx*4+2] < 100 &&
						 Math.min(numbers[i*4+3], numbers[cur_idx*4+3])-getMax(numbers[i*4+1], numbers[cur_idx*4+1]) > getMax(numbers[i*4+3]-numbers[i*4+1], numbers[cur_idx*4+3]-numbers[cur_idx*4+1])*0.8 ){
		
						 box_order_idx[idx++] = i;
						 cur_idx = i;
						 continue_while = 1;
						 checked_panel_idx[i] = 1;
						 break;
						 }
						 */
						if (Math.min(numbers.get(i).bottom, numbers.get(cur_idx).bottom) - Math.max(numbers.get(i).top, numbers.get(cur_idx).top) > Math.max(numbers.get(i).bottom - numbers.get(i).top, numbers.get(cur_idx).bottom - numbers.get(cur_idx).top) * 0.8) {
		
							if (numbers.get(i).left - numbers.get(cur_idx).right < min_hor_gap) {
								min_hor_gap = numbers.get(i).left - numbers.get(cur_idx).right; 
								min_hor_gap_idx = i;
							}
						}
		
					}
		
					if (min_hor_gap_idx >= 0) {
						box_order_idx[idx++] = min_hor_gap_idx;
						cur_idx = min_hor_gap_idx;
						continue_while = true;
						checked_panel_idx[min_hor_gap_idx] = 1;
						continue;
					}
		
					// new line or at the last box. 
					if (continue_while == false) {
		
						// check boxes with the left-most box in previous row.
						cur_idx = left_top_box_idx;
		
						j = 1000;
						first_col_box_idx = -1;
						for (i = 0; i < numbers.size(); i++) {
							if (cur_idx == i || checked_panel_idx[i] > 0)
								continue;
		
							// vertically close and horizontally overlapped 
							if (numbers.get(i).top - numbers.get(cur_idx).bottom < 100) { //&&
								//	Math.min(numbers[i*4+2], numbers[cur_idx*4+2])-getMax(numbers[i*4], numbers[cur_idx*4]) > getMax(numbers[i*4+2]-numbers[i*4], numbers[cur_idx*4+2]-numbers[cur_idx*4])*0.8 ){
								if (numbers.get(i).left < j) {
									j = numbers.get(i).left;
									first_col_box_idx = i;
								}
							}
						}
		
						// found a box. Otherwise, there's no more boxes. 
						if (first_col_box_idx != -1) {
							box_order_idx[idx++] = first_col_box_idx;
							cur_idx = first_col_box_idx;
							continue_while = true;
							checked_panel_idx[first_col_box_idx] = 1;
							left_top_box_idx = first_col_box_idx;
						}
					}
				}
		
				if( finalLabelSet.upper_or_lower_case == true )
					prev_box_label = 'a' - 1;
				else
					prev_box_label = 'A' - 1;
				
				for (i = 0; i < numbers.size(); i++) {
					// check out whether or not current box is labeled.
					// if so, store the label. 
		
					if (box_order_idx[i] < 0)
						continue;
						
					// box_order_idx has indices of ordered panels from left to right/top to bottom. So prev_box_label has index of the first unlabeled box.
					// this works if first box in the ordered idx has label but in case the first box has no label it causes problem. 
					// Suppose third box has label 'a', but since it has no label prev_box_label still is 'a'-1, and previous code just label the first box by 'a' that will be overwritten by already labeled third panel 'a'.
					// this cause less split boxes than image box detection result. 
					// fixed on Jan. 16, 2014.
					
					if (named_panel.get(box_order_idx[i]) > 0) {
						prev_box_label = named_panel_label.get(box_order_idx[i]);
						continue;
					}
		
					// prevent out of index error in accessing checked_label array.
					if( prev_box_label > 121 )
						prev_box_label = 121;
					
					if( prev_box_label > 89 && finalLabelSet.upper_or_lower_case == false )
						prev_box_label = 89;
												
					// find next unused labels.
					while ( (prev_box_label+1 >= 97 && checked_label[prev_box_label - 96] > 0) || (prev_box_label+1 < 97 && checked_label[prev_box_label - 64] > 0) ){
						prev_box_label++;
						
						if( prev_box_label > 121 || (prev_box_label < 97 && prev_box_label > 89) )
							break;
					}
					
					if( prev_box_label > 121 )
						prev_box_label = 121;
					
					if( prev_box_label > 89 && finalLabelSet.upper_or_lower_case == false )
						prev_box_label = 89;
					
					// if not labeled then label it with it's left neighbor's label. The next label of the neighbor's label.
					named_panel_label.set(box_order_idx[i], (char) (prev_box_label + 1));
					named_panel.set(box_order_idx[i], 1);
					if( prev_box_label+1 >= 97 )
						checked_label[prev_box_label-96] = 1;
					if( prev_box_label+1 < 97 )
						checked_label[prev_box_label-64] = 1;
					
		
					// if the assigned label is already used, then don't label current box.
					// e.g., panel A has two boxes and the algorithm detected two rectangle.
					// the second box has no label in it and no neighbor OCRed label.
					// first try to label it B.
					// But, in case B is already used to label the real panel B, then don't label current box.
					
				//	System.out.println("Prev: " + prev_box_label);
					
				//	if ( (prev_box_label+1 >= 97 && checked_label[prev_box_label - 96] > 0) || (prev_box_label+1 < 97 && checked_label[prev_box_label - 64] > 0) )
				//		named_panel.set(box_order_idx[i], 0);
		
					prev_box_label++;
					
				//	while( prev_box_label - 96 > 25 || prev_box_label - 64 > 25 )
				//		prev_box_label--;
					
					// to make sure that checked_label[prev_box_label -96 or 64] doesn't make any out of index error. Just label over 26 panels by Z or z (overwrite, maximum panels would be 26), according to upper_or_lower_case.
					if( prev_box_label > 121 )
						prev_box_label = 121;
					
					if( prev_box_label > 89 && finalLabelSet.upper_or_lower_case == false )
						prev_box_label = 89;
				}
			}
		
			for (i = 0; i < numbers.size(); i++) {
				if (split_box_idx.get(i) > 0 || named_panel.get(i) == 0)
					continue;
		
				if (numbers.get(i).left >= width || numbers.get(i).right >= width
						|| numbers.get(i).top >= height
						|| numbers.get(i).bottom >= height
						|| numbers.get(i).right <= numbers.get(i).left
						|| numbers.get(i).bottom <= numbers.get(i).top)
					continue;
		
				// split panels that have not been split from previous steps. 
				cvSetImageROI(inImg, cvRect(numbers.get(i).left,	numbers.get(i).top, Math.min(numbers.get(i).right - numbers.get(i).left + 1, inImg.width() - 1), Math.min(numbers.get(i).bottom	- numbers.get(i).top + 1, inImg.height() - 1)));
				
				IplImage cropped = cvCreateImage(cvSize(Math.min(numbers.get(i).right - numbers.get(i).left + 1, inImg.width() - 1), Math.min(numbers.get(i).bottom - numbers.get(i).top + 1, inImg.height() - 1)), inImg.depth(), inImg.nChannels());
				cvCopy(inImg, cropped, null);
		
				if( alphaStr.indexOf(named_panel_label.get(i)) != -1 ){
					final_label_list += named_panel_label.get(i);
					num_split_panels++;
				//	printf("save 4: %c\n", named_panel_label[i]);
					
					Final_Panel tmp_panel = new Final_Panel();
					tmp_panel.left = numbers.get(i).left;
					tmp_panel.top = numbers.get(i).top;
					tmp_panel.right = numbers.get(i).right;
					tmp_panel.bottom = numbers.get(i).bottom;
					tmp_panel.label = named_panel_label.get(i);
					
					finalSplitPanels.add(tmp_panel);
				}
		
				cvReleaseImage(cropped); cropped = null;
				cvResetImageROI(inImg);
		
				split_box_idx.set(i, 1);						
			}
		}
		}
		}
		
		cvReleaseImage(img); img = null;
		cvReleaseImage(inImg); inImg = null;
		
		if (resized_img != null){
		cvReleaseImage(resized_img); resized_img = null;
		}
		
		String returnStr = num_split_panels + ":::";
		
		for(i=0; i<num_split_panels; i++){
		if( i < num_split_panels-1 )
		returnStr += final_label_list.charAt(i) + ",";
		else
		returnStr += final_label_list.charAt(i);
		}
		
		/*
		String ground_fname = "";
		ground_fname = imgName;
		
		ground_fname = ground_fname.replaceAll("test_set", "xml");
		ground_fname = ground_fname.replaceAll(".jpg", ".xml");
		
		utilFunc.evaluatePanelSplittingResult(finalSplitPanels, ground_fname, evalNumbers);
		*/
		
		
		// output creation for ImageCLEF2013 multipanel separation task
		/*	
		FileWriter writerCLEF = new FileWriter("F:/DYou/workspace/iMEDLINE/samples/splitResult.xml", true);
		writerCLEF.write("   <annotation>" + "\n");
		writerCLEF.write("      <filename>" + fileName + "</filename>\n");
		
		for(i=0; i<finalSplitPanels.size(); i++){
		writerCLEF.write("      <object>\n");
		writerCLEF.write("         <point x=\"" + finalSplitPanels.get(i).left + "\"" + " y=\"" + finalSplitPanels.get(i).top + "\" />\n");
		writerCLEF.write("         <point x=\"" + finalSplitPanels.get(i).right + "\"" + " y=\"" + finalSplitPanels.get(i).top + "\" />\n");
		writerCLEF.write("         <point x=\"" + finalSplitPanels.get(i).left + "\"" + " y=\"" + finalSplitPanels.get(i).bottom + "\" />\n");
		writerCLEF.write("         <point x=\"" + finalSplitPanels.get(i).right + "\"" + " y=\"" + finalSplitPanels.get(i).bottom + "\" />\n");
		writerCLEF.write("      </object>\n");
		}
		
		writerCLEF.write("   </annotation>" + "\n");		
		writerCLEF.close();
		*/	
		return finalSplitPanels;
	}

	//added by Zhiyun (the output based on Daekeun's module only)
	public ArrayList<Final_Panel>  panelSplitter(String imgName, Classifier OCR_model){
		
		ArrayList<Final_Panel> finalSplitPanels = new ArrayList<Final_Panel>();
		
		IplImage inImg = cvLoadImage(imgName);
		
		//	cvSaveImage(out_fname, inImg);
		
		IplImage img, resized_img;
		ArrayList<PanelLabelDetection.PanelLabelInfo_Final> candLabelSet = new ArrayList<PanelLabelDetection.PanelLabelInfo_Final>();
		
		PanelLabelDetection.PanelLabelInfo_Final finalLabelSet = new PanelLabelDetection.PanelLabelInfo_Final();
		int num_split_panels = 0;
		
		int height = inImg.height();
		int width = inImg.width();
		

		// 200% enlargement for small image and color conversion
		resized_img = null;
		img = cvCreateImage(cvGetSize(inImg), IPL_DEPTH_8U, 1);

		if (inImg.nChannels() == 3 )
			cvCvtColor(inImg, img, CV_RGB2GRAY);			
		else
			cvCopy(inImg, img, null);

	//	Arrays.fill(candLabelSet, 0);
		
		PanelLabelDetection panelLabelDetectionRef = new PanelLabelDetection();
		
		
		// panel label detection. 
		// candLabelSet has all candidate label sets and the setNo denotes the total cand numbers. 
		if (width < 1000 && height < 1000) {
			resized_img = cvCreateImage(cvSize(width * 2, height * 2), IPL_DEPTH_8U, 1);
			cvResize(img, resized_img, 1);
			panelLabelDetectionRef.panelLabelDetection(resized_img, true, candLabelSet, OCR_model);			
		} else
			panelLabelDetectionRef.panelLabelDetection(img, false, candLabelSet, OCR_model);
		
		
		int setNo = panelLabelDetectionRef.setNo;	
		
		// remove duplicate rectangles across the candidate label sets
		for (int i = 0; i < candLabelSet.size(); i++){
			System.out.println("candidate label set: " + i);
			PanelLabelDetection.PanelLabelInfo_Final panelLabelInfo_Final = candLabelSet.get(i);
			ArrayList<PanelLabelDetection.PanelLabelInfo> labels = panelLabelInfo_Final.labels;
			for (int j = 0; j < labels.size(); j++ ){
				PanelLabelDetection.PanelLabelInfo label = labels.get(j);
				System.out.println("label: " + label.label + " left: "+ Integer.toString(label.left) + " right: "+ Integer.toString(label.right)
						+ " top: "+ Integer.toString(label.top) + " bottom: "+ Integer.toString(label.bottom));
				boolean isDuplicate = false;
				for (int k = 0; k < finalSplitPanels.size(); k++){
					if (finalSplitPanels.get(k).left == label.left && finalSplitPanels.get(k).right == label.right
							&& finalSplitPanels.get(k).top == label.top && finalSplitPanels.get(k).bottom == label.bottom){
						isDuplicate = true; 
						break;
					}
				}
				if (!isDuplicate){
					Final_Panel panel = new Final_Panel();
					panel.left = label.left;
					panel.right = label.right;
					panel.bottom = label.bottom;
					panel.top = label.top;
					finalSplitPanels.add(panel);
					
				}
				
			}
		}

		return finalSplitPanels;
	}

}