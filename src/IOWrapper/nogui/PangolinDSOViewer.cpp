/**
*/


#include "PangolinDSOViewer.h"
#include "KeyFrameDisplay.h"

#include "util/settings.h"
#include "util/globalCalib.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/ImmaturePoint.h"

namespace dso
{
namespace IOWrap
{

PangolinDSOViewer::PangolinDSOViewer(int w, int h, bool startRunThread)
{
	this->w = w;
	this->h = h;
	running = true;

	{
		boost::unique_lock<boost::mutex> lk(openImagesMutex);
		internalVideoImg = new MinimalImageB3(w,h);
		internalKFImg = new MinimalImageB3(w,h);
		internalResImg = new MinimalImageB3(w,h);
		videoImgChanged=kfImgChanged=resImgChanged=true;

		internalVideoImg->setBlack();
		internalKFImg->setBlack();
		internalResImg->setBlack();
	}


	{
		currentCam = new KeyFrameDisplay();
	}

	needReset = false;

    if(startRunThread)
        runThread = boost::thread(&PangolinDSOViewer::run, this);
}


PangolinDSOViewer::~PangolinDSOViewer()
{
	close();
	runThread.join();
}

void PangolinDSOViewer::printPC() {
	for(KeyFrameDisplay* fh : keyframes)
	{
        printf("kfd %d ", fh->id);
		fh->refreshPC(0, this->settings_scaledVarTH, this->settings_absVarTH,
			this->settings_pointCloudMode, this->settings_minRelBS, this->settings_sparsity);
	}
}

void PangolinDSOViewer::run()
{
	const int UI_WIDTH = 180;

	//while( running )
	{
		{
			// Activate efficiently by object
			boost::unique_lock<boost::mutex> lk3d(model3DMutex);
			int refreshed=0;
			for(KeyFrameDisplay* fh : keyframes)
			{
				float blue[3] = {0,0,1};
				if(this->settings_showKFCameras) fh->drawCam(1,blue,0.1);
                printf("%d ", fh->id);
				refreshed =+ (int)(fh->refreshPC(refreshed < 10, this->settings_scaledVarTH, this->settings_absVarTH,
						this->settings_pointCloudMode, this->settings_minRelBS, this->settings_sparsity));
				fh->drawPC(1);
			}
            printf("run-----\n");
			if(this->settings_showCurrentCamera) currentCam->drawCam(2,0,0.2);
			drawConstraints();
			lk3d.unlock();
		}

		openImagesMutex.lock();
		videoImgChanged=kfImgChanged=resImgChanged=false;
		openImagesMutex.unlock();

		{
			openImagesMutex.lock();
			float sd=0;
			for(float d : lastNMappingMs) sd+=d;
			openImagesMutex.unlock();
		}
		{
			model3DMutex.lock();
			float sd=0;
			for(float d : lastNTrackingMs) sd+=d;
			model3DMutex.unlock();
		}
	}

	printf("QUIT Pangolin thread!\n");
	//exit(1);
}


void PangolinDSOViewer::close()
{
	running = false;
}

void PangolinDSOViewer::join()
{
	runThread.join();
	printf("JOINED Pangolin thread!\n");
}

void PangolinDSOViewer::reset()
{
	needReset = true;
}

void PangolinDSOViewer::reset_internal()
{
	model3DMutex.lock();
	for(size_t i=0; i<keyframes.size();i++) delete keyframes[i];
	keyframes.clear();
	allFramePoses.clear();
	keyframesByKFID.clear();
	connections.clear();
	model3DMutex.unlock();

	openImagesMutex.lock();
	internalVideoImg->setBlack();
	internalKFImg->setBlack();
	internalResImg->setBlack();
	videoImgChanged= kfImgChanged= resImgChanged=true;
	openImagesMutex.unlock();

	needReset = false;
}


void PangolinDSOViewer::drawConstraints()
{
	if(settings_showAllConstraints)
	{
		// draw constraints
		for(unsigned int i=0;i<connections.size();i++)
		{
			if(connections[i].to == 0 || connections[i].from==0) continue;
			int nAct = connections[i].bwdAct + connections[i].fwdAct;
			int nMarg = connections[i].bwdMarg + connections[i].fwdMarg;
			if(nAct==0 && nMarg>0  )
			{
				Sophus::Vector3f t = connections[i].from->camToWorld.translation().cast<float>();
	//			glVertex3f((GLfloat) t[0],(GLfloat) t[1], (GLfloat) t[2]);
				t = connections[i].to->camToWorld.translation().cast<float>();
	//			glVertex3f((GLfloat) t[0],(GLfloat) t[1], (GLfloat) t[2]);
			}
		}
	}

	if(settings_showActiveConstraints)
	{
		for(unsigned int i=0;i<connections.size();i++)
		{
			if(connections[i].to == 0 || connections[i].from==0) continue;
			int nAct = connections[i].bwdAct + connections[i].fwdAct;

			if(nAct>0)
			{
				Sophus::Vector3f t = connections[i].from->camToWorld.translation().cast<float>();
	//			glVertex3f((GLfloat) t[0],(GLfloat) t[1], (GLfloat) t[2]);
				t = connections[i].to->camToWorld.translation().cast<float>();
	//			glVertex3f((GLfloat) t[0],(GLfloat) t[1], (GLfloat) t[2]);
			}
		}
	}

	if(settings_showTrajectory)
	{

		for(unsigned int i=0;i<keyframes.size();i++)
		{
	//		glVertex3f((float)keyframes[i]->camToWorld.translation()[0],
	//				(float)keyframes[i]->camToWorld.translation()[1],
	//				(float)keyframes[i]->camToWorld.translation()[2]);
		}
	}

	if(settings_showFullTrajectory)
	{

		for(unsigned int i=0;i<allFramePoses.size();i++)
		{
		//	glVertex3f((float)allFramePoses[i][0],
		//			(float)allFramePoses[i][1],
		//			(float)allFramePoses[i][2]);
		}
	}
}

void PangolinDSOViewer::publishGraph(const std::map<uint64_t,Eigen::Vector2i> &connectivity)
{
    if(!setting_render_display3D) return;
    if(disableAllDisplay) return;

	model3DMutex.lock();
    connections.resize(connectivity.size());
	int runningID=0;
	int totalActFwd=0, totalActBwd=0, totalMargFwd=0, totalMargBwd=0;
    for(std::pair<uint64_t,Eigen::Vector2i> p : connectivity)
	{
		int host = (int)(p.first >> 32);
        int target = (int)(p.first & (uint64_t)0xFFFFFFFF);

		assert(host >= 0 && target >= 0);
		if(host == target)
		{
			assert(p.second[0] == 0 && p.second[1] == 0);
			continue;
		}

		if(host > target) continue;

		connections[runningID].from = keyframesByKFID.count(host) == 0 ? 0 : keyframesByKFID[host];
		connections[runningID].to = keyframesByKFID.count(target) == 0 ? 0 : keyframesByKFID[target];
		connections[runningID].fwdAct = p.second[0];
		connections[runningID].fwdMarg = p.second[1];
		totalActFwd += p.second[0];
		totalMargFwd += p.second[1];

        uint64_t inverseKey = (((uint64_t)target) << 32) + ((uint64_t)host);
		Eigen::Vector2i st = connectivity.at(inverseKey);
		connections[runningID].bwdAct = st[0];
		connections[runningID].bwdMarg = st[1];

		totalActBwd += st[0];
		totalMargBwd += st[1];

		runningID++;
	}

	model3DMutex.unlock();
}

void PangolinDSOViewer::publishKeyframes(
		std::vector<FrameHessian*> &frames,
		bool final,
		CalibHessian* HCalib)
{
	//if(!setting_render_display3D) return;
    //if(disableAllDisplay) return;

	boost::unique_lock<boost::mutex> lk(model3DMutex);
	for(FrameHessian* fh : frames)
	{
        printf("%d ", fh->frameID);
		if(keyframesByKFID.find(fh->frameID) == keyframesByKFID.end())
		{
			KeyFrameDisplay* kfd = new KeyFrameDisplay();
			keyframesByKFID[fh->frameID] = kfd;
			keyframes.push_back(kfd);
		}
		keyframesByKFID[fh->frameID]->setFromKF(fh, HCalib);
	}
    printf("key-frame \n");
}

void PangolinDSOViewer::publishCamPose(FrameShell* frame,
		CalibHessian* HCalib)
{
    if(!setting_render_display3D) return;
    if(disableAllDisplay) return;

	boost::unique_lock<boost::mutex> lk(model3DMutex);
	struct timeval time_now;
	gettimeofday(&time_now, NULL);
	lastNTrackingMs.push_back(((time_now.tv_sec-last_track.tv_sec)*1000.0f + (time_now.tv_usec-last_track.tv_usec)/1000.0f));
	if(lastNTrackingMs.size() > 10) lastNTrackingMs.pop_front();
	last_track = time_now;

	if(!setting_render_display3D) return;

	currentCam->setFromF(frame, HCalib);
	allFramePoses.push_back(frame->camToWorld.translation().cast<float>());
}

void PangolinDSOViewer::pushLiveFrame(FrameHessian* image)
{
	if(!setting_render_displayVideo) return;
    if(disableAllDisplay) return;

	boost::unique_lock<boost::mutex> lk(openImagesMutex);

	for(int i=0;i<w*h;i++)
		internalVideoImg->data[i][0] =
		internalVideoImg->data[i][1] =
		internalVideoImg->data[i][2] =
			image->dI[i][0]*0.8 > 255.0f ? 255.0 : image->dI[i][0]*0.8;

	videoImgChanged=true;
}

bool PangolinDSOViewer::needPushDepthImage()
{
    return setting_render_displayDepth;
}

void PangolinDSOViewer::pushDepthImage(MinimalImageB3* image)
{

    if(!setting_render_displayDepth) return;
    if(disableAllDisplay) return;

	boost::unique_lock<boost::mutex> lk(openImagesMutex);

	struct timeval time_now;
	gettimeofday(&time_now, NULL);
	lastNMappingMs.push_back(((time_now.tv_sec-last_map.tv_sec)*1000.0f + (time_now.tv_usec-last_map.tv_usec)/1000.0f));
	if(lastNMappingMs.size() > 10) lastNMappingMs.pop_front();
	last_map = time_now;

	memcpy(internalKFImg->data, image->data, w*h*3);
	kfImgChanged=true;
}

}
}
