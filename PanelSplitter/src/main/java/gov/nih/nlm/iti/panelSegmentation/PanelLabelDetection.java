package gov.nih.nlm.iti.panelSegmentation;


/*
 * Update history
 * 
 * Jan. 17, 2014: the last two conditions for small characters filtering are commented out.
 * 
 * May 2, 2017: Jie Zou
 * 				upgrade to use mvn, JavaCV 1.3, and weka 3.6.14
 */

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.IplImage;
import org.bytedeco.javacpp.opencv_imgproc;
import static org.bytedeco.javacpp.opencv_core.*;

import java.util.ArrayList;
import java.util.Arrays;

import org.bytedeco.javacpp.Loader;

import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.classifiers.*;


public class PanelLabelDetection{

	public PanelLabelDetection(){}
	
	public int setNo = 0;
	
	private static final int NO_CLASS = 48;
	private static final int FEATURE_DIM = 36;
	private static boolean PROCESS_PMC_SET = true;
	private static double MSG_THRESHOLD	= 0.7;
	
	public class objRect{
		public int left;
		public int top;
		public int right;
		public int bottom;			
	}
	
	static public class PanelLabelInfo{
		int left, top, right, bottom;
		int cen_x, cen_y;
		char label;
		double score;
		boolean is_matched;
		double msg_0, msg_1;
		boolean checked;
		boolean used;	
	}

	public class OCR_result{
		int label;
		double score;		
	}
	
	
	static public class PanelLabelInfo_Final{
		ArrayList<PanelLabelInfo> labels;
		int noLabel;
		boolean upper_or_lower_case;
		int num_valid_label;
	}
	
	public class NeighborInfo{
		boolean hor_ver;
		int labelID;
		double msg_0, msg_1;	
	}

	public class Msg{
		double msg_0, msg_1;
	}
	
	
	String [] panel_alpha_set = {"A", "B", "Cc", "D", "E", "F", "G", "H", "I", "J", "Kk", "L", "M", "N", "Oo0", "Pp", "Q", "R", "Ss", "T", "Uu", "Vv", "Ww", "Xx", "Y", "Zz", "a", "b", "d", "e", "f", "g", "h", "ijl1", "m", "n", "q", "r", "t", "y", "2", "3", "4", "5", "6", "7", "8", "9"};

	UtilFunctions utilFunc = new UtilFunctions();
	
	public int panelLabelDetection(IplImage inImg, boolean resized, ArrayList<PanelLabelInfo_Final> finalLabelSet, Classifier OCR_model){
		IplImage binImg = null; 
		CvMemStorage storage = opencv_core.cvCreateMemStorage(0);
		
		CvSeq one_contour = new CvSeq(null);
		
		ArrayList<PanelLabelInfo> panelLabel = new ArrayList<PanelLabelInfo>();
		ArrayList<ArrayList<NeighborInfo>> neighbor = new ArrayList<ArrayList<NeighborInfo>>();
		ArrayList<ArrayList<Msg>> bp_msg = new ArrayList<ArrayList<Msg>>();
		ArrayList<ArrayList<Msg>> final_msg = new ArrayList<ArrayList<Msg>>();
		ArrayList<ArrayList<Integer>> support_neighbor = new ArrayList<ArrayList<Integer>>();
				
		int th_diff, setIdx, first_label_ascii;
		int panelLabelNo = 0;
		int i, j, k, m, n, x;
		int ret;
		boolean is_inner;		
		int noLabel, no_prior_label;
		
		int [] checked_prior_label = new int [26];
		int [] checked_panelLabel;
		int [] checked_alphabet = new int [26];
				
		double [] out = new double [NO_CLASS];
		double [] feature = new double [FEATURE_DIM];
		double area_ratio;
		double [] unary_evidence = null;
				
		binImg = opencv_core.cvCreateImage( opencv_core.cvGetSize(inImg), opencv_core.IPL_DEPTH_8U, 1 );
		
		// binarize twice to recognize black and white characters each time.
		for(x=0; x<4; x++){

			utilFunc.cvImgThreshold(inImg, binImg, 200);

			if( x == 1 ){
				utilFunc.cvImgThreshold(inImg, binImg, 50);
				utilFunc.negative_image(binImg);	
			}

			if( PROCESS_PMC_SET == true ){
				if( x == 2 ){
					utilFunc.cvImgThreshold(inImg, binImg, 128);
					utilFunc.negative_image(binImg);
				}
	
				if( x == 3 ){
					utilFunc.cvImgThreshold(inImg, binImg, 128);			
				}
			}
		
		//	cvSaveImage("F:\\test\\" + x + ".jpg", binImg);
			
		//	utilFunc.ShowImage(binImg, "Binimg");
		//	cvSaveImage("F:/test/bin.jpg", binImg);
			
		/*	if( x >= 4 ){
				if( x == 4 ){
					adpt_threshold->Create(width, height, 24, 0);
					convertOpenCV2CxImage3(inImg, adpt_threshold);
					adpt_threshold->AdaptiveThreshold(5, 64, 0, 0, 0.5);
					
					convertCxImage2OpenCV(adpt_threshold, binImg);
				}		

				if( x == 5 ){
					adpt_threshold->Negative();
					convertCxImage2OpenCV(adpt_threshold, binImg);
					delete adpt_threshold;
				}		
			}
		*/

		//	contours = null;
			
		//	System.out.println(x);
			CvSeq contours = new CvSeq(null);
			
			opencv_imgproc.cvFindContours(binImg, storage, contours, Loader.sizeof(CvContour.class), opencv_imgproc.CV_RETR_CCOMP, opencv_imgproc.CV_CHAIN_APPROX_NONE);

		//	IplImage contImg = opencv_core.cvCreateImage( opencv_core.cvGetSize(inImg), opencv_core.IPL_DEPTH_8U, 1 );
		
			while (contours != null && !contours.isNull()){

				// added on 12/20/2012
				// to prevent panelLabelNo exceeds 1000.
				if( panelLabelNo >= 999 )
					break;

				one_contour = contours;

				if( one_contour.total() < 30 ){
	                contours = contours.h_next();
	                continue;
				}
					
			//	utilFunc.drawContour(contImg, one_contour, 255);
				
				Arrays.fill(feature, 0);
				is_inner = false;
				
				objRect charPos = new objRect();
								
				ret = utilFunc.featureExtractionFromContour36(one_contour, feature, FEATURE_DIM, 0, charPos);
				is_inner = utilFunc.decisionMade;
				
				// the last two conditions are commented out on Jan. 17, 2014.
				if( charPos.right-charPos.left > 200 || charPos.bottom-charPos.top > 200 || (charPos.right-charPos.left) > (charPos.bottom-charPos.top)*2 /*|| (charPos.right-charPos.left) < 15 || (charPos.bottom - charPos.top) < 15*/ ){
					contours = contours.h_next();
					continue;
				}
										
				if( ret != -1 ){			
					Arrays.fill(out, 0);
										
					OCR_result ocrResult = new OCR_result();
										
					try {
						OCR(feature, ocrResult, OCR_model);
					} catch (Exception e) {						
						e.printStackTrace();
					}
					
				//	System.out.println("Ch " + panel_alpha_set[ocrResult.label] + ',' + ocrResult.score + ',' + charPos.left + ',' + charPos.top + ',' + charPos.right + ',' + charPos.bottom);
					
				//	utilFunc.drawContour(contImg, one_contour, 255);
					
					// digits. 
					if( ocrResult.label >= 40 ){
						contours = contours.h_next();
						continue;
					}

					// assume that only label from A~I or a~i are panel labels. 
				//	if( (max_label >= 9 && max_label <= 25) || max_label >= 34 )
				//		continue;

					// D can be recognized twice. The interior region D should not be stored. 
					if( ocrResult.label == 3 && is_inner == false ){
						contours = contours.h_next();
						continue;
					}
					
					PanelLabelInfo tmp_panelLabel = new PanelLabelInfo();
					tmp_panelLabel.left = charPos.left;
					tmp_panelLabel.top = charPos.top;
					tmp_panelLabel.right = charPos.right;
					tmp_panelLabel.bottom = charPos.bottom;
					tmp_panelLabel.cen_x = (charPos.left+charPos.right)/2;
					tmp_panelLabel.cen_y = (charPos.top+charPos.bottom)/2;
					tmp_panelLabel.score = ocrResult.score;
					tmp_panelLabel.label = panel_alpha_set[ocrResult.label].charAt(0);
					
					panelLabel.add(tmp_panelLabel);
					
					if( tmp_panelLabel.label == 'C' || tmp_panelLabel.label == 'K' || tmp_panelLabel.label == 'O' || tmp_panelLabel.label == 'P' || tmp_panelLabel.label == 'S' || tmp_panelLabel.label == 'U' || tmp_panelLabel.label == 'V' || tmp_panelLabel.label == 'W' || tmp_panelLabel.label == 'X' || tmp_panelLabel.label == 'Z'){
						PanelLabelInfo tmp_panelLabel2 = new PanelLabelInfo();
						
						tmp_panelLabel2.left = charPos.left;
						tmp_panelLabel2.top = charPos.top;
						tmp_panelLabel2.right = charPos.right;
						tmp_panelLabel2.bottom = charPos.bottom;
						tmp_panelLabel2.cen_x = (charPos.left+charPos.right)/2;
						tmp_panelLabel2.cen_y = (charPos.top+charPos.bottom)/2;
						tmp_panelLabel2.score = ocrResult.score;
						tmp_panelLabel2.label = panel_alpha_set[ocrResult.label].charAt(1);
						
						panelLabel.add(tmp_panelLabel2);
					}						
					else
						if( tmp_panelLabel.label == 'c' || tmp_panelLabel.label == 'k' || tmp_panelLabel.label == 'o' || tmp_panelLabel.label == 'p' || tmp_panelLabel.label == 's' || tmp_panelLabel.label == 'u' || tmp_panelLabel.label == 'v' || tmp_panelLabel.label == 'w' || tmp_panelLabel.label == 'x' || tmp_panelLabel.label == 'z'){
							PanelLabelInfo tmp_panelLabel2 = new PanelLabelInfo();
							
							tmp_panelLabel2.left = charPos.left;
							tmp_panelLabel2.top = charPos.top;
							tmp_panelLabel2.right = charPos.right;
							tmp_panelLabel2.bottom = charPos.bottom;
							tmp_panelLabel2.cen_x = (charPos.left+charPos.right)/2;
							tmp_panelLabel2.cen_y = (charPos.top+charPos.bottom)/2;
							tmp_panelLabel2.score = ocrResult.score;
							tmp_panelLabel2.label = panel_alpha_set[ocrResult.label].charAt(0);
							
							panelLabel.add(tmp_panelLabel2);					
							
						}
									
				//	if( panelLabelNo >= 999 )
					//	break;
				}
				
				contours = contours.h_next();
			}
			
		//	utilFunc.ShowImage(contImg, "ContourImage");
		//	cvSaveImage("F:\\test\\cont_" + x + ".jpg", contImg);
			
		//	cvReleaseImage(contImg); contImg = null;
		}
		
		panelLabelNo = panelLabel.size();
		
		if( resized == true )
			th_diff = 10;
		else
			th_diff = 5;

		if( panelLabel.size() > 0 ){
			int iter = 0;
			
			first_label_ascii = 0;
			
			// redundant OCR result removal.
			for(i=0; i<panelLabel.size(); i++){
				
				if( panelLabel.get(i).label == 0 )
					continue;

				for(j=0; j<panelLabel.size(); j++){
					if( i == j || panelLabel.get(j).label == 0 )
						continue;

					if( Math.abs(panelLabel.get(i).left - panelLabel.get(j).left) < th_diff && Math.abs(panelLabel.get(i).right - panelLabel.get(j).right) < th_diff &&
							Math.abs(panelLabel.get(i).top - panelLabel.get(j).top) < th_diff && Math.abs(panelLabel.get(i).bottom - panelLabel.get(j).bottom) < th_diff ){
						
						if( panelLabel.get(i).label == panelLabel.get(j).label ){
							if( panelLabel.get(i).score >= panelLabel.get(j).score )
								panelLabel.get(j).label = 0;
							else
							{
								panelLabel.get(i).label = 0;
								break;
							}
						}
					}
				}
			}
			
		//	System.out.println("Label 3: " + panelLabel.get(3).label);

			for(i=0; i<panelLabel.size(); i++){
				ArrayList<NeighborInfo> tmp_topArray = new ArrayList<NeighborInfo>();
				neighbor.add(tmp_topArray);				
			}
			
			
			for(iter = 0; iter < 2; iter++){

				if( iter == 0 )
					first_label_ascii = 65;		// detect upper case labels.
				else	
					first_label_ascii = 97;		// detect lower case labels.
								
			/*	if( iter == 1 ){
					for(i=0; i<panelLabel.size(); i++){
						if( panelLabel.get(i).label == 'C' || panelLabel.get(i).label == 'K' || panelLabel.get(i).label == 'O' || panelLabel.get(i).label == 'P' || panelLabel.get(i).label == 'S' || panelLabel.get(i).label == 'U' || panelLabel.get(i).label == 'V' || panelLabel.get(i).label == 'W' || panelLabel.get(i).label == 'X' || panelLabel.get(i).label == 'Z')
							panelLabel.get(i).label += 32; 
					}
				}
			*/	
				unary_evidence = new double [panelLabelNo];
				Arrays.fill(unary_evidence, 0);

				for(i=0; i<panelLabel.size(); i++){
					
					no_prior_label = 0;
					Arrays.fill(checked_prior_label, 0);
					
				/*	if( i == 382 ){
						System.out.println("382:" + iter + ":" + panelLabel.get(i).label);
						for(int kk=0; kk < neighbor.get(382).size(); kk++){
							System.out.println(panelLabel.get(neighbor.get(382).get(kk).labelID).label);
						}
					}
				*/	
					if( panelLabel.get(i).label == 0 )
						continue;
					
					if( (iter == 0 && panelLabel.get(i).label >= 97) || (iter == 1 && panelLabel.get(i).label < 97) )
						continue;
					
					for(j=0; j<panelLabel.size(); j++){
							
						NeighborInfo tmp_neighborInfo = new NeighborInfo();
						
						if( i == j || (iter == 0 && panelLabel.get(j).label >= 97) || (iter == 1 && panelLabel.get(j).label < 97) || panelLabel.get(j).label == 0 ){
							continue;
						}

						if( (panelLabel.get(j).right < panelLabel.get(i).left || panelLabel.get(j).bottom < panelLabel.get(i).top) && panelLabel.get(j).label < panelLabel.get(i).label ){
							if( checked_prior_label[Math.min(Math.max(panelLabel.get(j).label-first_label_ascii, 0), 25)] == 0 )
								checked_prior_label[Math.min(Math.max(panelLabel.get(j).label-first_label_ascii, 0), 25)] = 1;

						//	area_ratio = (getMin((panelLabel.get(i).right-panelLabel.get(i).left)*(panelLabel.get(i).bottom-panelLabel.get(i).top), (panelLabel.get(j).right-panelLabel.get(j).left)*(panelLabel.get(j).bottom-panelLabel.get(j).top))/(double)getMax((panelLabel.get(i).right-panelLabel.get(i).left)*(panelLabel.get(i).bottom-panelLabel.get(i).top), (panelLabel.get(j).right-panelLabel.get(j).left)*(panelLabel.get(j).bottom-panelLabel.get(j).top)));

						//	area_ratio >= 0.5 ? area_ratio = 1.0: area_ratio = 0;
						//	if( area_ratio > prior_label_area_ratio[panelLabel.get(j).label - first_label_ascii] )
						//		prior_label_area_ratio[panelLabel.get(j).label - first_label_ascii] = area_ratio;
						}
						
						if( panelLabel.get(j).cen_x > panelLabel.get(i).left && panelLabel.get(j).cen_x < panelLabel.get(i).right ){
							tmp_neighborInfo.hor_ver = true;
							tmp_neighborInfo.labelID = j;
							
							neighbor.get(i).add(tmp_neighborInfo);	
							
						//	if( iter == 0 && panelLabel.get(j).label >= 97  )
							//	System.out.println("Error:" + panelLabel.get(j).label);
						}
						
						// horizontal
						if( panelLabel.get(j).cen_y > panelLabel.get(i).top && panelLabel.get(j).cen_y < panelLabel.get(i).bottom ){
							tmp_neighborInfo.hor_ver = false;
							tmp_neighborInfo.labelID = j;		
							
							neighbor.get(i).add(tmp_neighborInfo);
							
						//	if( iter == 0 && panelLabel.get(j).label >= 97 )
							//	System.out.println("Error:" + panelLabel.get(j).label);
							
						//	tmp_topArray.add(tmp_neighborInfo);
						}
						
					}					
									
					area_ratio = 0;
					for(j=0; j<26; j++){
						if( checked_prior_label[j] > 0 )
							no_prior_label++;
						//	area_ratio += prior_label_area_ratio[j];
					}

					if( panelLabel.get(i).label > first_label_ascii + 2 )
						unary_evidence[i] = (0.6*Math.max((1-(panelLabel.get(i).label-first_label_ascii-no_prior_label)*0.1), 0.1)) + 0.4*panelLabel.get(i).score;
					else
					{
						// only A is set to its score for unary evidence. 
						// But, sometimes when A doesn't recognized correctly, algorithm can't find candidate with B or C since missing A
						// decrease the unary evidence due to small no_prior_label (0 for B or 1 for C)
						// for better tracking performance, set a maximum value to the evidence between score and unary evidence. 
						// Sometimes unary evidence is larger than score (recognition result)
						if( panelLabel.get(i).label == first_label_ascii )
							unary_evidence[i] = panelLabel.get(i).score;
						else
							unary_evidence[i] = Math.max(panelLabel.get(i).score, (0.6*no_prior_label/(double)(panelLabel.get(i).label-first_label_ascii) + 0.4*panelLabel.get(i).score));	// char A doesn't have any prior label.
					}
				}

				// message update
				{
					double binary_compatibility, pos_ratio;
					double from_0, from_1;
					double max_msg, scale_factor;
								
					// msg initialization
					for(i=0; i<panelLabel.size(); i++)
						for(j=0; j<neighbor.get(i).size(); j++)
							neighbor.get(i).get(j).msg_0 = neighbor.get(i).get(j).msg_1 = 1.0;

					// memory setup and initialize.
					for(i=0; i<panelLabel.size(); i++){
						
						ArrayList<Msg> tmp_msgArray = new ArrayList<Msg>();
						bp_msg.add(tmp_msgArray);
						
						for(j=0; j<neighbor.get(i).size(); j++){
							Msg tmp_oneArray = new Msg();
							
							bp_msg.get(i).add(tmp_oneArray);
						}
					}					
					
					
					for(i=0; i<panelLabel.size(); i++){
						
						ArrayList<Msg> tmp_msgArray = new ArrayList<Msg>();
						final_msg.add(tmp_msgArray);
						
						for(j=0; j<neighbor.get(i).size(); j++){
							Msg tmp_oneArray = new Msg();
							
							final_msg.get(i).add(tmp_oneArray);
						}
					}


					for(x = 0; x< 10; x++){
						max_msg = 0.0;

						for(i=0; i<panelLabel.size(); i++){
							
							for(j=0; j<neighbor.get(i).size(); j++){
								
								// eliminated neighbors.
								if( neighbor.get(i).get(j).labelID == -1 ){
									continue;
								}

								area_ratio = (Math.min((panelLabel.get(i).right-panelLabel.get(i).left)*(panelLabel.get(i).bottom-panelLabel.get(i).top), (panelLabel.get(neighbor.get(i).get(j).labelID).right-panelLabel.get(neighbor.get(i).get(j).labelID).left)*(panelLabel.get(neighbor.get(i).get(j).labelID).bottom-panelLabel.get(neighbor.get(i).get(j).labelID).top))/(double)Math.max((panelLabel.get(i).right-panelLabel.get(i).left)*(panelLabel.get(i).bottom-panelLabel.get(i).top), (panelLabel.get(neighbor.get(i).get(j).labelID).right-panelLabel.get(neighbor.get(i).get(j).labelID).left)*(panelLabel.get(neighbor.get(i).get(j).labelID).bottom-panelLabel.get(neighbor.get(i).get(j).labelID).top)));
							
								// vertical relationship
								if( neighbor.get(i).get(j).hor_ver == true ){
									if( (panelLabel.get(neighbor.get(i).get(j).labelID).bottom > panelLabel.get(i).bottom && panelLabel.get(neighbor.get(i).get(j).labelID).label > panelLabel.get(i).label)
										|| (panelLabel.get(neighbor.get(i).get(j).labelID).top < panelLabel.get(i).top && panelLabel.get(neighbor.get(i).get(j).labelID).label < panelLabel.get(i).label) )
										pos_ratio = 1.0;
									else
										pos_ratio = 0.0;									
								}
								else
								{
									if( (panelLabel.get(neighbor.get(i).get(j).labelID).right > panelLabel.get(i).right && panelLabel.get(neighbor.get(i).get(j).labelID).label > panelLabel.get(i).label)
										|| (panelLabel.get(neighbor.get(i).get(j).labelID).left < panelLabel.get(i).left && panelLabel.get(neighbor.get(i).get(j).labelID).label < panelLabel.get(i).label) )
										pos_ratio = 1.0;
									else
										pos_ratio = 0.0;
								}

								binary_compatibility = 0.4*area_ratio + 0.6*pos_ratio;
								
								from_0 = from_1 = 0.0;
								for(m=0; m<neighbor.get(neighbor.get(i).get(j).labelID).size(); m++){
									if( neighbor.get(neighbor.get(i).get(j).labelID).get(m).labelID == i )
										continue;

									if( from_0 < neighbor.get(neighbor.get(i).get(j).labelID).get(m).msg_0 )
										from_0 = neighbor.get(neighbor.get(i).get(j).labelID).get(m).msg_0;

									if( from_1 < neighbor.get(neighbor.get(i).get(j).labelID).get(m).msg_1 )
										from_1 = neighbor.get(neighbor.get(i).get(j).labelID).get(m).msg_1;
								}
								
								if( neighbor.get(neighbor.get(i).get(j).labelID).size() == 1 )
									from_0 = from_1 = 1.0;
						
								
								// msg update
								bp_msg.get(i).get(j).msg_0 = Math.max((1-unary_evidence[neighbor.get(i).get(j).labelID])*(1-binary_compatibility)*from_0, (unary_evidence[neighbor.get(i).get(j).labelID])*(1-binary_compatibility)*from_1);
								bp_msg.get(i).get(j).msg_1 = Math.max((1-unary_evidence[neighbor.get(i).get(j).labelID])*(1-binary_compatibility)*from_0, (unary_evidence[neighbor.get(i).get(j).labelID])*(binary_compatibility)*from_1);

								bp_msg.get(i).get(j).msg_0 = Math.max(bp_msg.get(i).get(j).msg_0, 0.0000001);
								bp_msg.get(i).get(j).msg_1 = Math.max(bp_msg.get(i).get(j).msg_1, 0.0000001);

								if( Math.max(bp_msg.get(i).get(j).msg_0, bp_msg.get(i).get(j).msg_1) > max_msg )
									max_msg = Math.max(bp_msg.get(i).get(j).msg_0, bp_msg.get(i).get(j).msg_1);							
							}						
						}
						
						// normalize msgs so that there's no underflow. 
						if( max_msg < 0.5 )
							scale_factor = 0.9/max_msg;
						else
							scale_factor = 1.0;

						for(i=0; i<panelLabel.size(); i++){
							for(j=0; j<neighbor.get(i).size(); j++){

								if( neighbor.get(i).get(j).labelID == -1 )
									continue;

								neighbor.get(i).get(j).msg_0 = bp_msg.get(i).get(j).msg_0*scale_factor;
								neighbor.get(i).get(j).msg_1 = bp_msg.get(i).get(j).msg_1*scale_factor;
							}
						}

						if( x == 0 ){
							for(i=0; i<panelLabel.size(); i++){
								for(j=0; j<neighbor.get(i).size(); j++){
									
									if( neighbor.get(i).get(j).labelID == -1 ){
										continue;
									}
																		
									final_msg.get(i).get(j).msg_0 = neighbor.get(i).get(j).msg_0;
									final_msg.get(i).get(j).msg_1 = neighbor.get(i).get(j).msg_1;								
								}
							}
						}
						else
						{
							for(i=0; i<panelLabel.size(); i++){
								
								for(j=0; j<neighbor.get(i).size(); j++){
									if( neighbor.get(i).get(j).labelID == -1 ){
										continue;
									}

									if( (final_msg.get(i).get(j).msg_0 > final_msg.get(i).get(j).msg_1 && neighbor.get(i).get(j).msg_0 < neighbor.get(i).get(j).msg_1) ||
										(final_msg.get(i).get(j).msg_0 < final_msg.get(i).get(j).msg_1 && neighbor.get(i).get(j).msg_0 > neighbor.get(i).get(j).msg_1) ){
										final_msg.get(i).get(j).msg_0 = neighbor.get(i).get(j).msg_0;
										final_msg.get(i).get(j).msg_1 = neighbor.get(i).get(j).msg_1;
									}
																	
								}								
							}
						}
					}
				}
				
				{
					double max_0, max_1;

					for(i=0; i<panelLabel.size(); i++){
						if( panelLabel.get(i).label == 0 )
							continue;

						max_0 = max_1 = 0.0;
						for(j=0; j<neighbor.get(i).size(); j++){
							if( panelLabel.get(neighbor.get(i).get(j).labelID).label == 0 || neighbor.get(i).get(j).labelID == -1 )
								continue;

							if( final_msg.get(i).get(j).msg_0 > max_0 )
								max_0 = final_msg.get(i).get(j).msg_0;
				
							if( final_msg.get(i).get(j).msg_1 > max_1 )
								max_1 = final_msg.get(i).get(j).msg_1;
						}
						
						panelLabel.get(i).msg_0 = max_0*(1-unary_evidence[i]);
						panelLabel.get(i).msg_1 = max_1*unary_evidence[i];
					}
				}
					
				
				{
					double max_msg_A = 0;
					int max_msg_A_idx = -1;
					int idx, alphabet_idx, min_alphabet_idx;
					int [] done_alphabet = new int [26];
					int min_nei_dist;
					char being_checked_alphahet;
					double max_4_one_alphabet;
					int max_4_one_alphabet_idx;

					checked_panelLabel = new int [panelLabelNo];
					
					while ( true )
					{
						Arrays.fill(checked_panelLabel, 0);

						idx = alphabet_idx = min_alphabet_idx = max_4_one_alphabet_idx = -1;
						max_4_one_alphabet = 0.0;					
						
						// try to find A, B, or C to start.
						// find an A with the biggest msg.
						// eventually, this search all A~E, from the biggest A to the smallest E.
						// and search cand label sets from each starting label.
						for(j=0; j<5; j++){
							max_msg_A = MSG_THRESHOLD;
							max_msg_A_idx = -1;

							for(i=0; i<panelLabelNo; i++){
								if( panelLabel.get(i).label == first_label_ascii + j && panelLabel.get(i).msg_1 > max_msg_A && panelLabel.get(i).checked == false ){
									max_msg_A = panelLabel.get(i).msg_1;
									max_msg_A_idx = i;										
								}
							}

							if( max_msg_A_idx != -1 ){
								idx = j;
								panelLabel.get(max_msg_A_idx).checked = true;
								break;
							}
						}
						
						// below arrays should be initialized whenever a new label set is searched from a label.
						for(i=0; i<26; i++){
							ArrayList<Integer> tmp_support = new ArrayList<Integer>();
							
							support_neighbor.add(tmp_support);							
						}						
					
						Arrays.fill(checked_alphabet, 0);

						// found a starting label.
						if( max_msg_A_idx > -1 ){
							Arrays.fill(done_alphabet, 0);
							// search neighbor labels that has high msg1.
							for(j=0; j<neighbor.get(max_msg_A_idx).size(); j++){
								if( neighbor.get(max_msg_A_idx).get(j).labelID == -1 )
									continue;
								
							//	System.out.println(panelLabel.get(neighbor.get(max_msg_A_idx).get(j).labelID).label);
							//	System.out.println("Started:" + max_msg_A_idx);
							//	for(int kk=0; kk < neighbor.get(max_msg_A_idx).size(); kk++){
							//		System.out.println(panelLabel.get(neighbor.get(max_msg_A_idx).get(kk).labelID).label);
							//	}
								
								if( final_msg.get(max_msg_A_idx).get(j).msg_1 > MSG_THRESHOLD && done_alphabet[Math.min(Math.max(0, panelLabel.get(neighbor.get(max_msg_A_idx).get(j).labelID).label-first_label_ascii), 25)] == 0 ){

								//	support_neighbor[idx][support_neighbor_no[idx]++] = neighbor[max_msg_A_idx][j].labelID;

									being_checked_alphahet = panelLabel.get(neighbor.get(max_msg_A_idx).get(j).labelID).label;
									max_4_one_alphabet = final_msg.get(max_msg_A_idx).get(j).msg_1;
									max_4_one_alphabet_idx = j;
									
									for(k=0; k<neighbor.get(max_msg_A_idx).size(); k++){
										if( k == j )
											continue;

										if( panelLabel.get(neighbor.get(max_msg_A_idx).get(k).labelID).label == being_checked_alphahet && final_msg.get(max_msg_A_idx).get(k).msg_1 > max_4_one_alphabet ){
											max_4_one_alphabet = final_msg.get(max_msg_A_idx).get(k).msg_1;
											max_4_one_alphabet_idx = k;
										}
									}

									if( max_4_one_alphabet_idx != -1 ){
										support_neighbor.get(idx).add(neighbor.get(max_msg_A_idx).get(max_4_one_alphabet_idx).labelID);
									}

									// uncomment this and delete all others in this if statement to go back to the code for ICDAR.
								//	support_neighbor[idx][support_neighbor_no[idx]++] = neighbor[support_neighbor[idx][i]][j].labelID;

									done_alphabet[Math.min(Math.max(0, panelLabel.get(neighbor.get(max_msg_A_idx).get(j).labelID).label-first_label_ascii), 25)] = 1;
								}
							}
							
							checked_panelLabel[max_msg_A_idx] = 1;
							
							// track neighbor labels of neighboring labels of the starting label.
							// if checked_alphabet[idx] == 2, then the alphabet and its neighbor labels are checked. 
							checked_alphabet[Math.min(Math.max(idx, 0), 25)] = 2;
							while( idx != 100 ){
								min_alphabet_idx = 100;

								for(i=0; i<support_neighbor.get(idx).size(); i++){
									if( panelLabel.get(support_neighbor.get(idx).get(i)).msg_1 > MSG_THRESHOLD && checked_panelLabel[support_neighbor.get(idx).get(i)] == 0 ){

										checked_panelLabel[support_neighbor.get(idx).get(i)] = 1;
										alphabet_idx = panelLabel.get(support_neighbor.get(idx).get(i)).label-first_label_ascii;

										if( support_neighbor.get(alphabet_idx).size() > 0 )
											continue;

										if( checked_alphabet[Math.min(Math.max(0, alphabet_idx), 25)] == 0 )
											checked_alphabet[Math.min(Math.max(0, alphabet_idx), 25)] = 1;

										Arrays.fill(done_alphabet, 0);
										for(j=0; j<neighbor.get(support_neighbor.get(idx).get(i)).size(); j++){
											if( neighbor.get(support_neighbor.get(idx).get(i)).get(j).labelID == -1 )
												continue;

											if( final_msg.get(support_neighbor.get(idx).get(i)).get(j).msg_1 > MSG_THRESHOLD && done_alphabet[Math.min(Math.max(0, panelLabel.get(neighbor.get(support_neighbor.get(idx).get(i)).get(j).labelID).label-first_label_ascii), 25)] == 0 ){
												
												// added to solve problems in 225593.jpg
												// panel label B has lots of neighbor O having high score and they are all added to the support neighbors of B. 
												// But, due to the code if( support_neighbor_no[alphabet_idx] ) statement, only one O that has added at first is considered. 
												// Hence, have to find only one O that propagate maximum message to B and check it. 
												// The original code aslo check only one O. 
												being_checked_alphahet = panelLabel.get(neighbor.get(support_neighbor.get(idx).get(i)).get(j).labelID).label;
												max_4_one_alphabet = final_msg.get(support_neighbor.get(idx).get(i)).get(j).msg_1;
												max_4_one_alphabet_idx = j;
												for(k=0; k<neighbor.get(support_neighbor.get(idx).get(i)).size(); k++){

													if( k == j || neighbor.get(support_neighbor.get(idx).get(i)).get(k).labelID == -1 )
														continue;

													if( panelLabel.get(neighbor.get(support_neighbor.get(idx).get(i)).get(k).labelID).label == being_checked_alphahet && final_msg.get(support_neighbor.get(idx).get(i)).get(k).msg_1 > max_4_one_alphabet ){
														max_4_one_alphabet = final_msg.get(support_neighbor.get(idx).get(i)).get(k).msg_1;
														max_4_one_alphabet_idx = k;
													}
												}

												if( max_4_one_alphabet_idx != -1 )
													support_neighbor.get(alphabet_idx).add(neighbor.get(support_neighbor.get(idx).get(i)).get(max_4_one_alphabet_idx).labelID);

												// uncomment this and delete all others in this if statement to go back to the code for ICDAR.
											//	support_neighbor[idx][support_neighbor_no[idx]++] = neighbor[support_neighbor[idx][i]][j].labelID;

												done_alphabet[Math.min(Math.max(0, panelLabel.get(neighbor.get(support_neighbor.get(idx).get(i)).get(j).labelID).label-first_label_ascii), 25)] = 1;
											}
										}

										if( alphabet_idx < min_alphabet_idx )
											min_alphabet_idx = alphabet_idx;
									}
								}

								// set up the next search label.
								idx = min_alphabet_idx;
								
								// to solve a problem happening in 106585.jpg
								if( min_alphabet_idx != 100 ){
									checked_alphabet[Math.min(Math.max(0, min_alphabet_idx), 25)] = 2;
								}
								else
								{
									for(i=0; i<26; i++){
										if( checked_alphabet[i] == 1 ){
											idx = i;
											checked_alphabet[idx] = 2;
											break;
										}
									}
								}
							}

						/*	for(i=1; i<25; i++){
								// both neighboring alphabets are found but not the middle one due to several reasons. 
								if( checked_alphabet[i-1] == 2 || checked_alphabet[i+1] == 2 ){
									for(j=0; j<panelLabelNo; j++){
										if( panelLabel.get(j).msg_1 > 0.7 && panelLabel.get(j).label == i+first_label_ascii ){
											detected_nei_no = 0;
											for(m=0; m<neighbor_no[j]; m++){
												if( final_msg[j][m].msg_1 > 0.9 && (checked_panelLabel[neighbor[j][m].labelID] || panelLabel[neighbor[j][m].labelID].msg_1 > 0.8) )
													detected_nei_no++;
											}

											if( detected_nei_no >= 2 ){
												checked_panelLabel[j] = 1;
												break;
											}
										}
									}
								}
							}
						*/
							// Add a text label (panelLabel[j]) if 
							// it has some neighbors that detected as panel label and its ascii distance is smaller than 3.
							for(j=0; j<panelLabel.size(); j++){
								min_nei_dist = 100;
								if( panelLabel.get(j).label == 0 )
									continue;

								if( panelLabel.get(j).msg_1 > 0.7 && panelLabel.get(j).msg_0 < 0.1 && checked_panelLabel[j] == 0 && checked_alphabet[Math.min(Math.max(0, panelLabel.get(j).label-first_label_ascii), 25)] == 0 ){
									for(i=0; i<neighbor.get(j).size(); i++){

										if( neighbor.get(j).get(i).labelID == -1 )
											continue;

										if( final_msg.get(j).get(i).msg_1 > MSG_THRESHOLD && checked_panelLabel[neighbor.get(j).get(i).labelID] > 0 )
											if( Math.abs(panelLabel.get(neighbor.get(j).get(i).labelID).label-panelLabel.get(j).label) < min_nei_dist )
												min_nei_dist = Math.abs(panelLabel.get(neighbor.get(j).get(i).labelID).label-panelLabel.get(j).label);
									}
									
									if( min_nei_dist <= 3 ){
										checked_panelLabel[j] = 1;
									// comment out after ICDAR test
									// no need to break. it prevents other labels to be checked. 
									//	break;
									}
								}
							}

							noLabel = 0;
							PanelLabelInfo_Final tmp_candSet = new PanelLabelInfo_Final();
							tmp_candSet.labels = new ArrayList<PanelLabelInfo>(); 
														
							for(i=0; i<panelLabelNo; i++){
								if( checked_panelLabel[i] == 0 || panelLabel.get(i).label == 0 )
									continue;

							//	if( panelLabel.get(i).msg_1 < 0.8 )
							//		continue;

								PanelLabelInfo tmp_panelLabelInfo = new PanelLabelInfo();							
								
								// save the labels in the set. 
								if( resized == true ){
									tmp_panelLabelInfo.left = panelLabel.get(i).left/2;
									tmp_panelLabelInfo.right = panelLabel.get(i).right/2;
									tmp_panelLabelInfo.top = panelLabel.get(i).top/2;
									tmp_panelLabelInfo.bottom = panelLabel.get(i).bottom/2;

									tmp_panelLabelInfo.cen_x = panelLabel.get(i).cen_x/2;
									tmp_panelLabelInfo.cen_y = panelLabel.get(i).cen_y/2;
								}
								else
								{
									tmp_panelLabelInfo.left = panelLabel.get(i).left;
									tmp_panelLabelInfo.right = panelLabel.get(i).right;
									tmp_panelLabelInfo.top = panelLabel.get(i).top;
									tmp_panelLabelInfo.bottom = panelLabel.get(i).bottom;

									tmp_panelLabelInfo.cen_x = panelLabel.get(i).cen_x;
									tmp_panelLabelInfo.cen_y = panelLabel.get(i).cen_y;
								}
															
								tmp_panelLabelInfo.score = panelLabel.get(i).score;
								tmp_panelLabelInfo.label = panelLabel.get(i).label;
								
							//	System.out.println("Label: " + panelLabel.get(i).label);
								
								
								tmp_candSet.labels.add(tmp_panelLabelInfo);
								
							}

							if( tmp_candSet.labels.size() > 0 ){
								tmp_candSet.noLabel = tmp_candSet.labels.size();

								if( iter == 0 )
									tmp_candSet.upper_or_lower_case = false;
								else
									tmp_candSet.upper_or_lower_case = true;

								finalLabelSet.add(tmp_candSet);								
							}

						}
						else	// break while loop
							break;					
					}

					checked_panelLabel = null;
				}
		
				unary_evidence = null;		
			}
		}

		opencv_core.cvReleaseImage(binImg);

		// eliminate idenitcal sets.
		for(i=0; i<finalLabelSet.size(); i++){
			if( finalLabelSet.get(i).noLabel == 0 )
				continue;

			for(j=0; j<finalLabelSet.size(); j++){
				if( i == j )
					continue;

				if( finalLabelSet.get(i).noLabel == finalLabelSet.get(j).noLabel ){
					if( finalLabelSet.get(i).labels.equals(finalLabelSet.get(j).labels) )
						finalLabelSet.get(j).noLabel = 0;
				}
			}
		}

		neighbor.clear();
		bp_msg.clear();
		final_msg.clear();
		
		
	/*	System.out.println(finalLabelSet.size());
		
		for(setIdx = 0; setIdx<finalLabelSet.size(); setIdx++){
			for(i=0; i<finalLabelSet.get(setIdx).labels.size(); i++)
				System.out.println(finalLabelSet.get(setIdx).labels.get(i).label);
		}
	*/
		
		for(setIdx=0; setIdx<finalLabelSet.size(); setIdx++){
			
			if( finalLabelSet.get(setIdx).noLabel > 0 && finalLabelSet.get(setIdx).labels.get(0).label < 97 )
				first_label_ascii = 65;		
			else	
				first_label_ascii = 97;		
			
			neighbor.clear();
			bp_msg.clear();
			final_msg.clear();
			
			panelLabelNo = finalLabelSet.get(setIdx).noLabel;
			
			for(i=0; i<panelLabelNo; i++){
				ArrayList<NeighborInfo> tmp_topArray = new ArrayList<NeighborInfo>();
				neighbor.add(tmp_topArray);				
			}
						
			unary_evidence = new double [panelLabelNo];
			Arrays.fill(unary_evidence, 0);

			for(i=0; i<panelLabelNo; i++){
				
				no_prior_label = 0;
				Arrays.fill(checked_prior_label, 0);
				
				if( finalLabelSet.get(setIdx).labels.get(i).label == 0 )
					continue;
				
				for(j=0; j<panelLabelNo; j++){
					NeighborInfo tmp_neighborInfo = new NeighborInfo();
					
					if( i == j || finalLabelSet.get(setIdx).labels.get(i).label == 0 )
						continue;
					
					if( (finalLabelSet.get(setIdx).labels.get(j).right < finalLabelSet.get(setIdx).labels.get(i).left || finalLabelSet.get(setIdx).labels.get(j).bottom < finalLabelSet.get(setIdx).labels.get(i).top) && finalLabelSet.get(setIdx).labels.get(j).label < finalLabelSet.get(setIdx).labels.get(i).label ){
					//	System.out.println(finalLabelSet.get(setIdx).labels.get(j).label + ":" + first_label_ascii);
						if( checked_prior_label[Math.min(Math.max(finalLabelSet.get(setIdx).labels.get(j).label - first_label_ascii, 0), 25)] == 0 )
							checked_prior_label[Math.min(Math.max(finalLabelSet.get(setIdx).labels.get(j).label - first_label_ascii, 0), 25)] = 1;

					//	area_ratio = (getMin((panelLabel.get(i).right-panelLabel.get(i).left)*(panelLabel.get(i).bottom-panelLabel.get(i).top), (finalLabelSet.get(setIdx).labels.get(j).right-finalLabelSet.get(setIdx).labels.get(j).left)*(finalLabelSet.get(setIdx).labels.get(j).bottom-finalLabelSet.get(setIdx).labels.get(j).top))/(double)getMax((panelLabel.get(i).right-panelLabel.get(i).left)*(panelLabel.get(i).bottom-panelLabel.get(i).top), (finalLabelSet.get(setIdx).labels.get(j).right-finalLabelSet.get(setIdx).labels.get(j).left)*(finalLabelSet.get(setIdx).labels.get(j).bottom-finalLabelSet.get(setIdx).labels.get(j).top)));

					//	area_ratio >= 0.5 ? area_ratio = 1.0: area_ratio = 0;
					//	if( area_ratio > prior_label_area_ratio[panelLabel.get(j).label - first_label_ascii] )
					//		prior_label_area_ratio[panelLabel.get(j).label - first_label_ascii] = area_ratio;
					}
					
				//	if( finalLabelSet.get(setIdx).labels.get(j).cen_x > finalLabelSet.get(setIdx).labels.get(i).left && finalLabelSet.get(setIdx).labels.get(j).cen_x < finalLabelSet.get(setIdx).labels.get(i).right ){
					if( Math.min(finalLabelSet.get(setIdx).labels.get(j).right, finalLabelSet.get(setIdx).labels.get(i).right) - Math.max(finalLabelSet.get(setIdx).labels.get(j).left, finalLabelSet.get(setIdx).labels.get(i).left) > 0 ){
						tmp_neighborInfo.hor_ver = true;
						tmp_neighborInfo.labelID = j;
						
						neighbor.get(i).add(tmp_neighborInfo);						
					}

				// horizontal
				//	if( finalLabelSet.get(setIdx).labels.get(j).cen_y > finalLabelSet.get(setIdx).labels.get(i).top && finalLabelSet.get(setIdx).labels.get(j).cen_y < finalLabelSet.get(setIdx).labels.get(i).bottom ){
					if( Math.min(finalLabelSet.get(setIdx).labels.get(j).bottom, finalLabelSet.get(setIdx).labels.get(i).bottom) - Math.max(finalLabelSet.get(setIdx).labels.get(j).top, finalLabelSet.get(setIdx).labels.get(i).top) > 0 ){
						tmp_neighborInfo.hor_ver = false;
						tmp_neighborInfo.labelID = j;
						
						neighbor.get(i).add(tmp_neighborInfo);
					}				
									
				}
				
				area_ratio = 0;
				for(j=0; j<26; j++){
					if( checked_prior_label[j] > 0 )
						no_prior_label++;
					//	area_ratio += prior_label_area_ratio[j];
				}

				if( finalLabelSet.get(setIdx).labels.get(i).label > first_label_ascii )
					unary_evidence[i] = (0.6*Math.max((1-(finalLabelSet.get(setIdx).labels.get(i).label-first_label_ascii-no_prior_label)*0.1), 0.1) + 0.4*finalLabelSet.get(setIdx).labels.get(i).score);	
				else
					unary_evidence[i] = finalLabelSet.get(setIdx).labels.get(i).score;	// char A doesn't have any prior label.
			}

			// message update
			{
				double binary_compatibility, pos_ratio;
				double from_0, from_1;
				double max_msg, scale_factor;
						
				// msg initialization
				for(i=0; i<panelLabelNo; i++)
					for(j=0; j<neighbor.get(i).size(); j++)
						neighbor.get(i).get(j).msg_0 = neighbor.get(i).get(j).msg_1 = 1.0;
				
				for(i=0; i<panelLabelNo; i++){
					
					ArrayList<Msg> tmp_msgArray = new ArrayList<Msg>();
					bp_msg.add(tmp_msgArray);
					
					for(j=0; j<neighbor.get(i).size(); j++){
						Msg tmp_oneArray = new Msg();
						
						bp_msg.get(i).add(tmp_oneArray);
					}
				}					
				
				
				for(i=0; i<panelLabelNo; i++){
					
					ArrayList<Msg> tmp_msgArray = new ArrayList<Msg>();
					final_msg.add(tmp_msgArray);
					
					for(j=0; j<neighbor.get(i).size(); j++){
						Msg tmp_oneArray = new Msg();
						
						final_msg.get(i).add(tmp_oneArray);
					}
				}				
				
				for(x = 0; x < 10; x++){
					max_msg = 0.0;

					for(i=0; i<panelLabelNo; i++){
						
						for(j=0; j<neighbor.get(i).size(); j++){
							// eliminated neighbors.
							if( neighbor.get(i).get(j).labelID == -1 )
								continue;
							
							area_ratio = (Math.min((finalLabelSet.get(setIdx).labels.get(i).right-finalLabelSet.get(setIdx).labels.get(i).left)*(finalLabelSet.get(setIdx).labels.get(i).bottom-finalLabelSet.get(setIdx).labels.get(i).top), (finalLabelSet.get(setIdx).labels.get(neighbor.get(i).get(j).labelID).right-finalLabelSet.get(setIdx).labels.get(neighbor.get(i).get(j).labelID).left)*(finalLabelSet.get(setIdx).labels.get(neighbor.get(i).get(j).labelID).bottom-finalLabelSet.get(setIdx).labels.get(neighbor.get(i).get(j).labelID).top))/(double)Math.max((finalLabelSet.get(setIdx).labels.get(i).right-finalLabelSet.get(setIdx).labels.get(i).left)*(finalLabelSet.get(setIdx).labels.get(i).bottom-finalLabelSet.get(setIdx).labels.get(i).top), (finalLabelSet.get(setIdx).labels.get(neighbor.get(i).get(j).labelID).right-finalLabelSet.get(setIdx).labels.get(neighbor.get(i).get(j).labelID).left)*(finalLabelSet.get(setIdx).labels.get(neighbor.get(i).get(j).labelID).bottom-finalLabelSet.get(setIdx).labels.get(neighbor.get(i).get(j).labelID).top)));
						
							// vertical relationship
							if( neighbor.get(i).get(j).hor_ver == true ){
								if( (finalLabelSet.get(setIdx).labels.get(neighbor.get(i).get(j).labelID).bottom > finalLabelSet.get(setIdx).labels.get(i).bottom && finalLabelSet.get(setIdx).labels.get(neighbor.get(i).get(j).labelID).label > finalLabelSet.get(setIdx).labels.get(i).label)
									|| (finalLabelSet.get(setIdx).labels.get(neighbor.get(i).get(j).labelID).top < finalLabelSet.get(setIdx).labels.get(i).top && finalLabelSet.get(setIdx).labels.get(neighbor.get(i).get(j).labelID).label < finalLabelSet.get(setIdx).labels.get(i).label) )
									pos_ratio = 1.0;
								else
									pos_ratio = 0.0;
							}
							else
							{
								if( (finalLabelSet.get(setIdx).labels.get(neighbor.get(i).get(j).labelID).right > finalLabelSet.get(setIdx).labels.get(i).right && finalLabelSet.get(setIdx).labels.get(neighbor.get(i).get(j).labelID).label > finalLabelSet.get(setIdx).labels.get(i).label)
									|| (finalLabelSet.get(setIdx).labels.get(neighbor.get(i).get(j).labelID).left < finalLabelSet.get(setIdx).labels.get(i).left && finalLabelSet.get(setIdx).labels.get(neighbor.get(i).get(j).labelID).label < finalLabelSet.get(setIdx).labels.get(i).label) )
									pos_ratio = 1.0;
								else
									pos_ratio = 0.0;
							}

							// the neighbors are already aligned and hence the area_ratio is more important factor than pos_ratio.
							binary_compatibility = 0.6*area_ratio + 0.4*pos_ratio;
							
							from_0 = from_1 = 0.0;
							for(m=0; m<neighbor.get(neighbor.get(i).get(j).labelID).size(); m++){
								if( neighbor.get(neighbor.get(i).get(j).labelID).get(m).labelID == i )
									continue;

								if( from_0 < neighbor.get(neighbor.get(i).get(j).labelID).get(m).msg_0 )
									from_0 = neighbor.get(neighbor.get(i).get(j).labelID).get(m).msg_0;

								if( from_1 < neighbor.get(neighbor.get(i).get(j).labelID).get(m).msg_1 )
									from_1 = neighbor.get(neighbor.get(i).get(j).labelID).get(m).msg_1;
							}
							
							if( neighbor.get(neighbor.get(i).get(j).labelID).size() == 1 )
								from_0 = from_1 = 1.0;
														
							// msg update
							bp_msg.get(i).get(j).msg_0 = Math.max((1-unary_evidence[neighbor.get(i).get(j).labelID])*(1-binary_compatibility)*from_0, (unary_evidence[neighbor.get(i).get(j).labelID])*(1-binary_compatibility)*from_1);
							bp_msg.get(i).get(j).msg_1 = Math.max((1-unary_evidence[neighbor.get(i).get(j).labelID])*(1-binary_compatibility)*from_0, (unary_evidence[neighbor.get(i).get(j).labelID])*(binary_compatibility)*from_1);

							bp_msg.get(i).get(j).msg_0 = Math.max(bp_msg.get(i).get(j).msg_0, 0.0000001);
							bp_msg.get(i).get(j).msg_1 = Math.max(bp_msg.get(i).get(j).msg_1, 0.0000001);

							if( Math.max(bp_msg.get(i).get(j).msg_0, bp_msg.get(i).get(j).msg_1) > max_msg )
								max_msg = Math.max(bp_msg.get(i).get(j).msg_0, bp_msg.get(i).get(j).msg_1);						
							
						}					
					}
					
					// normalize msgs so that there's no underflow. 
					if( max_msg < 0.5 )
						scale_factor = 0.9/max_msg;
					else
						scale_factor = 1.0;

					for(i=0; i<panelLabelNo; i++){
						for(j=0; j<neighbor.get(i).size(); j++){

							if( neighbor.get(i).get(j).labelID == -1 )
								continue;

							neighbor.get(i).get(j).msg_0 = bp_msg.get(i).get(j).msg_0*scale_factor;
							neighbor.get(i).get(j).msg_1 = bp_msg.get(i).get(j).msg_1*scale_factor;
						}
					}

					if( x == 0 ){
						for(i=0; i<panelLabelNo; i++){
							for(j=0; j<neighbor.get(i).size(); j++){
							
								if( neighbor.get(i).get(j).labelID == -1 )
									continue;

								final_msg.get(i).get(j).msg_0 = neighbor.get(i).get(j).msg_0;
								final_msg.get(i).get(j).msg_1 = neighbor.get(i).get(j).msg_1;																
							}
						}
					}
					else
					{
						for(i=0; i<panelLabelNo; i++){
							for(j=0; j<neighbor.get(i).size(); j++){

								if( neighbor.get(i).get(j).labelID == -1 )
									continue;

								if( (final_msg.get(i).get(j).msg_0 > final_msg.get(i).get(j).msg_1 && neighbor.get(i).get(j).msg_0 < neighbor.get(i).get(j).msg_1) ||
									(final_msg.get(i).get(j).msg_0 < final_msg.get(i).get(j).msg_1 && neighbor.get(i).get(j).msg_0 > neighbor.get(i).get(j).msg_1) ){
									final_msg.get(i).get(j).msg_0 = neighbor.get(i).get(j).msg_0;
									final_msg.get(i).get(j).msg_1 = neighbor.get(i).get(j).msg_1;
								}
							}
						}
					}
				}
			}
			
			{
				double max_0, max_1;
				
				for(i=0; i<panelLabelNo; i++){
					if( finalLabelSet.get(setIdx).labels.get(i).label == 0 )
						continue;

					max_0 = max_1 = 0.0;
					
					for(j=0; j<neighbor.get(i).size(); j++){
						if( finalLabelSet.get(setIdx).labels.get(neighbor.get(i).get(j).labelID).label == 0 || neighbor.get(i).get(j).labelID == -1 )
							continue;

						if( final_msg.get(i).get(j).msg_0 > max_0 )
							max_0 = final_msg.get(i).get(j).msg_0;
			
						if( final_msg.get(i).get(j).msg_1 > max_1 )
							max_1 = final_msg.get(i).get(j).msg_1;
					}
					
					finalLabelSet.get(setIdx).labels.get(i).msg_0 = max_0*(1-unary_evidence[i]);
					finalLabelSet.get(setIdx).labels.get(i).msg_1 = max_1*unary_evidence[i];				
				}
			}

			{
				double max_msg_A = 0;
				int max_msg_A_idx = -1;
				int idx, alphabet_idx, min_alphabet_idx;
				int [] done_alphabet = new int [26];
				int min_nei_dist;
				char being_checked_alphahet;
				double max_4_one_alphabet;
				int max_4_one_alphabet_idx;
				int []  best_checked_panelLabel;
				
				int max_match_no = 0;

				checked_panelLabel = new int [panelLabelNo];
				best_checked_panelLabel = new int [panelLabelNo];
				
				Arrays.fill(best_checked_panelLabel, 0);
				Arrays.fill(checked_panelLabel, 0);
			//	while ( 1 )
				{					
					Arrays.fill(checked_panelLabel, 0);
					
					idx = alphabet_idx = min_alphabet_idx = max_4_one_alphabet_idx = -1;
					max_4_one_alphabet = 0.0;					
					
					// try to find A, B, or C to start.
					// find an A with the biggest msg.
					// eventually, this search all A~E, from the biggest A to the smallest E.
					// and search cand label sets from each starting label.
					for(j=0; j<5; j++){
						max_msg_A = MSG_THRESHOLD;
						max_msg_A_idx = -1;

						for(i=0; i<panelLabelNo; i++){
							if( finalLabelSet.get(setIdx).labels.get(i).label == first_label_ascii + j && finalLabelSet.get(setIdx).labels.get(i).msg_1 > max_msg_A && finalLabelSet.get(setIdx).labels.get(i).checked == false ){
								max_msg_A = finalLabelSet.get(setIdx).labels.get(i).msg_1;
								max_msg_A_idx = i;										
							}
						}

						if( max_msg_A_idx != -1 ){
							idx = j;
							finalLabelSet.get(setIdx).labels.get(max_msg_A_idx).checked = false;
							break;
						}
					}
					
					// below arrays should be initialized whenever a new label set is searched from a label.
					support_neighbor.clear();
					for(i=0; i<26; i++){
						ArrayList<Integer> tmp_support = new ArrayList<Integer>();
						support_neighbor.add(tmp_support);		
					}

					Arrays.fill(checked_alphabet, 0);

					// found a starting label.
					if( max_msg_A_idx > -1 ){
						Arrays.fill(done_alphabet, 0);
						// search neighbor labels that has high msg1.
						for(j=0; j<neighbor.get(max_msg_A_idx).size(); j++){
							if( neighbor.get(max_msg_A_idx).get(j).labelID == -1 )
								continue;

							if( final_msg.get(max_msg_A_idx).get(j).msg_1 > MSG_THRESHOLD && done_alphabet[Math.min(Math.max(0,finalLabelSet.get(setIdx).labels.get(neighbor.get(max_msg_A_idx).get(j).labelID).label-first_label_ascii), 25)] == 0 ){

							//	support_neighbor[idx][support_neighbor_no[idx]++] = neighbor[max_msg_A_idx][j].labelID;

								being_checked_alphahet = finalLabelSet.get(setIdx).labels.get(neighbor.get(max_msg_A_idx).get(j).labelID).label;
								max_4_one_alphabet = final_msg.get(max_msg_A_idx).get(j).msg_1;
								max_4_one_alphabet_idx = j;
								
								for(k=0; k<neighbor.get(max_msg_A_idx).size(); k++){
									if( k == j )
										continue;

									if( finalLabelSet.get(setIdx).labels.get(neighbor.get(max_msg_A_idx).get(k).labelID).label == being_checked_alphahet && final_msg.get(max_msg_A_idx).get(k).msg_1 > max_4_one_alphabet ){
										max_4_one_alphabet = final_msg.get(max_msg_A_idx).get(k).msg_1;
										max_4_one_alphabet_idx = k;
									}
								}

								if( max_4_one_alphabet_idx != -1 ){
									support_neighbor.get(idx).add(neighbor.get(max_msg_A_idx).get(max_4_one_alphabet_idx).labelID);
								}

								// uncomment this and delete all others in this if statement to go back to the code for ICDAR.
							//	support_neighbor[idx][support_neighbor_no[idx]++] = neighbor[support_neighbor[idx][i]][j].labelID;

								done_alphabet[Math.min(Math.max(0, finalLabelSet.get(setIdx).labels.get(neighbor.get(max_msg_A_idx).get(j).labelID).label-first_label_ascii), 25)] = 1;
							}
						}
						
						checked_panelLabel[max_msg_A_idx] = 1;
						
						// track neighbor labels of neighboring labels of the starting label.
						// if checked_alphabet[idx] == 2, then the alphabet and its neighbor labels are checked. 
						checked_alphabet[Math.min(Math.max(0, idx), 25)] = 2;
						while( idx != 100 ){
							min_alphabet_idx = 100;
							for(i=0; i<support_neighbor.get(idx).size(); i++){
								if( finalLabelSet.get(setIdx).labels.get(support_neighbor.get(idx).get(i)).msg_1 > MSG_THRESHOLD && checked_panelLabel[support_neighbor.get(idx).get(i)] == 0 ){

									checked_panelLabel[support_neighbor.get(idx).get(i)] = 1;
									alphabet_idx = finalLabelSet.get(setIdx).labels.get(support_neighbor.get(idx).get(i)).label-first_label_ascii;

									if( support_neighbor.get(alphabet_idx).size() > 0 )
										continue;

									if( checked_alphabet[Math.min(Math.max(0, alphabet_idx), 25)] == 0 )
										checked_alphabet[Math.min(Math.max(0, alphabet_idx), 25)] = 1;

									Arrays.fill(done_alphabet, 0);
									for(j=0; j<neighbor.get(support_neighbor.get(idx).get(i)).size(); j++){
										if( neighbor.get(support_neighbor.get(idx).get(i)).get(j).labelID == -1 )
											continue;

										if( final_msg.get(support_neighbor.get(idx).get(i)).get(j).msg_1 > MSG_THRESHOLD && done_alphabet[Math.min(Math.max(0, finalLabelSet.get(setIdx).labels.get(neighbor.get(support_neighbor.get(idx).get(i)).get(j).labelID).label-first_label_ascii), 25)] == 0 ){
											
											// added to solve problems in 225593.jpg
											// panel label B has lots of neighbor O having high score and they are all added to the support neighbors of B. 
											// But, due to the code if( support_neighbor_no[alphabet_idx] ) statement, only one O that has added at first is considered. 
											// Hence, have to find only one O that propagate maximum message to B and check it. 
											// The original code aslo check only one O. 
											being_checked_alphahet = finalLabelSet.get(setIdx).labels.get(neighbor.get(support_neighbor.get(idx).get(i)).get(j).labelID).label;
											max_4_one_alphabet = final_msg.get(support_neighbor.get(idx).get(i)).get(j).msg_1;
											max_4_one_alphabet_idx = j;
											for(k=0; k<neighbor.get(support_neighbor.get(idx).get(i)).size(); k++){

												if( k == j || neighbor.get(support_neighbor.get(idx).get(i)).get(k).labelID == -1 )
													continue;

												if( finalLabelSet.get(setIdx).labels.get(neighbor.get(support_neighbor.get(idx).get(i)).get(k).labelID).label == being_checked_alphahet && final_msg.get(support_neighbor.get(idx).get(i)).get(k).msg_1 > max_4_one_alphabet ){
													max_4_one_alphabet = final_msg.get(support_neighbor.get(idx).get(i)).get(k).msg_1;
													max_4_one_alphabet_idx = k;
												}
											}

											if( max_4_one_alphabet_idx != -1 )
												support_neighbor.get(alphabet_idx).add(neighbor.get(support_neighbor.get(idx).get(i)).get(max_4_one_alphabet_idx).labelID);

											// uncomment this and delete all others in this if statement to go back to the code for ICDAR.
										//	support_neighbor[idx][support_neighbor_no[idx]++] = neighbor[support_neighbor[idx][i]][j].labelID;

											done_alphabet[Math.min(Math.max(0, finalLabelSet.get(setIdx).labels.get(neighbor.get(support_neighbor.get(idx).get(i)).get(j).labelID).label-first_label_ascii), 25)] = 1;
										}
									}

									if( alphabet_idx < min_alphabet_idx )
										min_alphabet_idx = alphabet_idx;
								}
							}

							// set up the next search label.
							idx = min_alphabet_idx;
							
							// to solve a problem happening in 106585.jpg
							if( min_alphabet_idx != 100 ){
								checked_alphabet[Math.min(Math.max(0, min_alphabet_idx), 25)] = 2;
							}
							else
							{
								for(i=0; i<26; i++){
									if( checked_alphabet[i] == 1 ){
										idx = i;
										checked_alphabet[idx] = 2;
										break;
									}
								}
							}
						}

						// Add a text label (panelLabel[j]) if 
						// it has some neighbors that detected as panel label and its ascii distance is smaller than 3.
						for(j=0; j<panelLabelNo; j++){
							min_nei_dist = 100;
							if( finalLabelSet.get(setIdx).labels.get(j).label == 0 )
								continue;

							if( finalLabelSet.get(setIdx).labels.get(j).msg_1 > 0.7 && finalLabelSet.get(setIdx).labels.get(j).msg_0 < 0.1 && checked_panelLabel[j] == 0 && checked_alphabet[Math.min(Math.max(0, finalLabelSet.get(setIdx).labels.get(j).label-first_label_ascii), 25)] == 0 ){
								for(i=0; i<neighbor.get(j).size(); i++){

									if( neighbor.get(j).get(i).labelID == -1 )
										continue;

									if( final_msg.get(j).get(i).msg_1 > MSG_THRESHOLD && checked_panelLabel[neighbor.get(j).get(i).labelID] > 0 )
										if( Math.abs(finalLabelSet.get(setIdx).labels.get(neighbor.get(j).get(i).labelID).label-finalLabelSet.get(setIdx).labels.get(j).label) < min_nei_dist )
											min_nei_dist = Math.abs(finalLabelSet.get(setIdx).labels.get(neighbor.get(j).get(i).labelID).label-finalLabelSet.get(setIdx).labels.get(j).label);
								}
								
								if( min_nei_dist <= 3 )
									checked_panelLabel[j] = 1;							
							}
						}

						noLabel = 0;
						for(i=0; i<panelLabelNo; i++){
							if( checked_panelLabel[i] > 0 )//|| finalLabelSet.get(setIdx).labels.get(i).msg_1 < 0.8 )
							//	finalLabelSet.get(setIdx).labels.get(i).label = 0;						
							noLabel++;
						}

						if( noLabel > max_match_no ){
							max_match_no = noLabel;
							for(i=0; i<panelLabelNo; i++)
								best_checked_panelLabel[i] = checked_panelLabel[i];							
						}

						noLabel = 0;
					}
					else
						break;
				}

				if( max_match_no > 0 ){
					for(i=0; i<panelLabelNo; i++){
						if( best_checked_panelLabel[i] == 0 )
							finalLabelSet.get(setIdx).labels.get(i).label = 0;
					}
				}

			//	checked_panelLabel = null;
			//	best_checked_panelLabel = null;
			}
			
			// implemented on Dec/02/2011 to remove several noise labels identical with a true label in a set.
			// check out each label's neighbor and count the distance.
			// compare the distance of two identical label and leave one that has more closest labels. 
			{
				int [][] neighbor_dist;
				int nei_label_dist;
				
				neighbor_dist = new int [panelLabelNo][26];
				
				for(i=0; i<panelLabelNo; i++){
					if( finalLabelSet.get(setIdx).labels.get(i).label == 0 )
						continue;

					for(j=0; j<neighbor.get(i).size(); j++){
						nei_label_dist = Math.abs(finalLabelSet.get(setIdx).labels.get(neighbor.get(i).get(j).labelID).label - finalLabelSet.get(setIdx).labels.get(i).label);
						
						if( nei_label_dist < 26 )
							neighbor_dist[i][nei_label_dist]++;
					}				
				}			
				
				for(i=0; i<panelLabelNo; i++){
					if( finalLabelSet.get(setIdx).labels.get(i).label == 0 )
						continue;

					for(j=0; j<panelLabelNo; j++){

						if( finalLabelSet.get(setIdx).labels.get(j).label == 0 || finalLabelSet.get(setIdx).labels.get(j).label != finalLabelSet.get(setIdx).labels.get(i).label )
							continue;

						for(k=1; k<26; k++){
							if( neighbor_dist[i][k] == neighbor_dist[j][k] )
								continue;

							if( neighbor_dist[i][k] > neighbor_dist[j][k] ){
								finalLabelSet.get(setIdx).labels.get(j).label = 0;
								break;
							}
							else
							{
								finalLabelSet.get(setIdx).labels.get(i).label = 0;
								break;
							}
						}
					}
				}

			//	neighbor_dist = null;
			}

		//	neighbor = null;
		//	bp_msg = null;
		//	final_msg = null;
	
		//	unary_evidence = null;		
		}

		opencv_core.cvReleaseMemStorage(storage);	


		for(k=0; k<finalLabelSet.size(); k++){
			for(i=0; i<finalLabelSet.get(k).noLabel; i++){
				
				if( finalLabelSet.get(k).labels.get(i).label == 0 )
					continue;

				for(j=0; j<finalLabelSet.get(k).noLabel; j++){
					if( i == j || finalLabelSet.get(k).labels.get(j).label == 0 )
						continue;

					// overlapped CCs. leave only one with high msg1
					if( Math.min(finalLabelSet.get(k).labels.get(i).right, finalLabelSet.get(k).labels.get(j).right) - Math.max(finalLabelSet.get(k).labels.get(i).left, finalLabelSet.get(k).labels.get(j).left) > 0 && 
						Math.min(finalLabelSet.get(k).labels.get(i).bottom, finalLabelSet.get(k).labels.get(j).bottom) - Math.max(finalLabelSet.get(k).labels.get(i).top, finalLabelSet.get(k).labels.get(j).top) > 0 ){
											
						if( finalLabelSet.get(k).labels.get(i).msg_1 >= finalLabelSet.get(k).labels.get(j).msg_1 )
							finalLabelSet.get(k).labels.get(j).label = 0;
						else
						{
							finalLabelSet.get(k).labels.get(i).label = 0;
							break;
						}					
					}
				}
			}
		}


		int valid_labels;

		for(setIdx=0; setIdx < finalLabelSet.size(); setIdx++){

			valid_labels = 0;

			for(i=0; i<finalLabelSet.get(setIdx).noLabel; i++){
				if( finalLabelSet.get(setIdx).labels.get(i).label == 0 )
					continue;
				valid_labels++;
			}
			finalLabelSet.get(setIdx).num_valid_label = valid_labels;
		}

		// Remove identical results.

		int match_label;

		for(i=0; i<finalLabelSet.size(); i++){
			if( finalLabelSet.get(i).noLabel == 0 )
				continue;

			for(j=0; j<finalLabelSet.size(); j++){
				if( i == j || finalLabelSet.get(j).noLabel == 0 || finalLabelSet.get(i).num_valid_label != finalLabelSet.get(j).num_valid_label )
					continue;

				match_label = 0;

				for(m = 0; m<finalLabelSet.get(i).noLabel; m++){
					if( finalLabelSet.get(i).labels.get(m).label == 0 )
						continue;

					for(n=0; n<finalLabelSet.get(j).noLabel; n++){
						if( finalLabelSet.get(j).labels.get(n).label == 0 )
							continue;

						if( finalLabelSet.get(i).labels.get(m).label == finalLabelSet.get(j).labels.get(n).label &&
							finalLabelSet.get(i).labels.get(m).cen_x == finalLabelSet.get(j).labels.get(n).cen_x &&
							finalLabelSet.get(i).labels.get(m).cen_y == finalLabelSet.get(j).labels.get(n).cen_y ){

							match_label++;
							break;
						}
					}
				}

				if( match_label == finalLabelSet.get(i).num_valid_label )
					finalLabelSet.get(j).noLabel = 0;
			}

		}

		this.setNo = finalLabelSet.size();

		return 1;
	}

	
	private void OCR(double[] feature, OCR_result ocrResult, Classifier OCR) throws Exception {
		String attrStr = "";
				
		FastVector atts = new FastVector();
		
		for(int i=0; i<36; i++){
			attrStr = "att" + i;
			atts.addElement(new Attribute(attrStr));
		}
		 
		atts.addElement(new Attribute("att36", (FastVector) null));
		Instances data = new Instances("ocr", atts, 0);
		
		double[] vals = new double[data.numAttributes()];
		
		for(int i=0; i<36; i++){
			vals[i] = feature[i];			
		}
		
		vals[36] = data.attribute(36).addStringValue("");
		data.add(new Instance(1.0, vals));
		
		data.instance(0).setMissing(36);
		
	//	Instances Unlabeled_Instance = new Instances( new BufferedReader(new FileReader("F:/DYou/workspace/PanelSplitting_Java/singleLetter.arff")));  
		
		// set class attribute
	//	Unlabeled_Instance.setClassIndex(Unlabeled_Instance.numAttributes() - 1);
		data.setClassIndex(data.numAttributes() - 1);
		
		// create copy
	//	Instances Labeled_Instance = new Instances( new BufferedReader(new FileReader("F:/CLEF_CLASSIFIER/Labeled_Instance.arff")));
	//	Labeled_Instance.setClassIndex(Labeled_Instance.numAttributes() - 1);
		
	//	File fileOpen =new File("F:/CLEF_CLASSIFIER/result_flat.txt");
				
		// label instances
		double clsLabel = OCR.classifyInstance(data.instance(0));			
		
		double [] dist = OCR.distributionForInstance(data.instance(0));
		
		ocrResult.score = dist[(int)clsLabel];
		ocrResult.label = (int)clsLabel;
		
	}

	
	
	
}