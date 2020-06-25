/* CUDA 10.2 has a bug that prevents including <cuda/atomic> from two separate
 * object files. As a workaround, we include ex2.cu directly here. */
#include "ex2.cu"

#include <cassert>
#include <iostream>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include <infiniband/verbs.h>

class server_rpc_context : public rdma_server_context {
private:
    std::unique_ptr<gpu_image_processing_context> gpu_context;

public:
    explicit server_rpc_context(uint16_t tcp_port) : rdma_server_context(tcp_port),
        gpu_context(create_queues_server(256))
    {
    }

    virtual void event_loop() override
    {
        /* so the protocol goes like this:
         * 1. we'll wait for a CQE indicating that we got an Send request from the client.
         *    this tells us we have new work to do. The wr_id we used in post_recv tells us
         *    where the request is.
         * 2. now we send an RDMA Read to the client to retrieve the request.
         *    we will get a completion indicating the read has completed.
         * 3. we process the request on the GPU.
         * 4. upon completion, we send an RDMA Write with immediate to the client with
         *    the results.
         */
        rpc_request* req;
        uchar *img_in;
        uchar *img_out;

        bool terminate = false, got_last_cqe = false;

        while (!terminate || !got_last_cqe) {
            // Step 1: Poll for CQE
            struct ibv_wc wc;
            int ncqes = ibv_poll_cq(cq, 1, &wc);
            if (ncqes < 0) {
                perror("ibv_poll_cq() failed");
                exit(1);
            }
            if (ncqes > 0) {
		VERBS_WC_CHECK(wc);

                switch (wc.opcode) {
                case IBV_WC_RECV:
                    /* Received a new request from the client */
                    req = &requests[wc.wr_id];
                    img_in = &images_in[wc.wr_id * IMG_SZ];

                    /* Terminate signal */
                    if (req->request_id == -1) {
                        printf("Terminating...\n");
                        terminate = true;
                        goto send_rdma_write;
                    }

                    /* Step 2: send RDMA Read to client to read the input */
                    post_rdma_read(
                        img_in,             // local_src
                        req->input_length,  // len
                        mr_images_in->lkey, // lkey
                        req->input_addr,    // remote_dst
                        req->input_rkey,    // rkey
                        wc.wr_id);          // wr_id
                    break;

                case IBV_WC_RDMA_READ:
                    /* Completed RDMA read for a request */
                    req = &requests[wc.wr_id];
                    img_in = &images_in[wc.wr_id * IMG_SZ];
                    img_out = &images_out[wc.wr_id * IMG_SZ];

                    // Step 3: Process on GPU
                    gpu_context->enqueue(wc.wr_id, img_in, img_out);
					break;
                    
                case IBV_WC_RDMA_WRITE:
                    /* Completed RDMA Write - reuse buffers for receiving the next requests */
                    post_recv(wc.wr_id % OUTSTANDING_REQUESTS);

		    if (terminate)
			got_last_cqe = true;

                    break;
                default:
                    printf("Unexpected completion\n");
                    assert(false);
                }
            }

            // Dequeue completed GPU tasks
            int dequeued_img_id;
            if (gpu_context->dequeue(&dequeued_img_id)) {
                req = &requests[dequeued_img_id];
                img_out = &images_out[dequeued_img_id * IMG_SZ];

send_rdma_write:
                // Step 4: Send RDMA Write with immediate to client with the response
				post_rdma_write(
                    req->output_addr,                       // remote_dst
                    terminate ? 0 : req->output_length,     // len
                    req->output_rkey,                       // rkey
                    terminate ? 0 : img_out,                // local_src
                    mr_images_out->lkey,                    // lkey
                    dequeued_img_id + OUTSTANDING_REQUESTS, // wr_id
                    (uint32_t*)&req->request_id);           // immediate
            }
        }
    }
};

class client_rpc_context : public rdma_client_context {
private:
    uint32_t requests_sent = 0;
    uint32_t send_cqes_received = 0;

    struct ibv_mr *mr_images_in; /* Memory region for input images */
    struct ibv_mr *mr_images_out; /* Memory region for output images */
public:
    explicit client_rpc_context(uint16_t tcp_port) : rdma_client_context(tcp_port)
    {
    }

    ~client_rpc_context()
    {
        kill();
    }

    virtual void set_input_images(uchar *images_in, size_t bytes) override
    {
        /* register a memory region for the input images. */
        mr_images_in = ibv_reg_mr(pd, images_in, bytes, IBV_ACCESS_REMOTE_READ);
        if (!mr_images_in) {
            perror("ibv_reg_mr() failed for input images");
            exit(1);
        }
    }

    virtual void set_output_images(uchar *images_out, size_t bytes) override
    {
        /* register a memory region for the output images. */
        mr_images_out = ibv_reg_mr(pd, images_out, bytes, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        if (!mr_images_out) {
            perror("ibv_reg_mr() failed for output images");
            exit(1);
        }
    }

    virtual bool enqueue(int img_id, uchar *img_in, uchar *img_out) override
    {
        if (requests_sent - send_cqes_received == OUTSTANDING_REQUESTS)
            return false;

        struct ibv_sge sg; /* scatter/gather element */
        struct ibv_send_wr wr; /* WQE */
        struct ibv_send_wr *bad_wr; /* ibv_post_send() reports bad WQEs here */

        /* step 1: send request to server using Send operation */
        
        struct rpc_request *req = &requests[requests_sent % OUTSTANDING_REQUESTS];
        req->request_id = img_id;
        req->input_rkey = img_in ? mr_images_in->rkey : 0;
        req->input_addr = (uintptr_t)img_in;
        req->input_length = IMG_SZ;
        req->output_rkey = img_out ? mr_images_out->rkey : 0;
        req->output_addr = (uintptr_t)img_out;
        req->output_length = IMG_SZ;

        /* RDMA send needs a gather element (local buffer)*/
        memset(&sg, 0, sizeof(struct ibv_sge));
        sg.addr = (uintptr_t)req;
        sg.length = sizeof(*req);
        sg.lkey = mr_requests->lkey;

        /* WQE */
        memset(&wr, 0, sizeof(struct ibv_send_wr));
        wr.wr_id = img_id; /* helps identify the WQE */
        wr.sg_list = &sg;
        wr.num_sge = 1;
        wr.opcode = IBV_WR_SEND;
        wr.send_flags = IBV_SEND_SIGNALED; /* always set this in this excersize. generates CQE */

        /* post the WQE to the HCA to execute it */
        if (ibv_post_send(qp, &wr, &bad_wr)) {
            perror("ibv_post_send() failed");
            exit(1);
        }

        ++requests_sent;

        return true;
    }

    virtual bool dequeue(int *img_id) override
    {
        /* When WQE is completed we expect a CQE */
        /* We also expect a completion of the RDMA Write with immediate operation from the server to us */
        /* The order between the two is not guarenteed */

        struct ibv_wc wc; /* CQE */
        int ncqes = ibv_poll_cq(cq, 1, &wc);
        if (ncqes < 0) {
            perror("ibv_poll_cq() failed");
            exit(1);
        }
        if (ncqes == 0)
            return false;

	VERBS_WC_CHECK(wc);

        switch (wc.opcode) {
        case IBV_WC_SEND:
            ++send_cqes_received;
            return false;
        case IBV_WC_RECV_RDMA_WITH_IMM:
            *img_id = wc.imm_data;
            break;
        default:
            printf("Unexpected completion type\n");
            assert(0);
        }

        /* step 2: post receive buffer for the next RPC call (next RDMA write with imm) */
        post_recv();

        return true;
    }

    void kill()
    {
        while (!enqueue(-1, // Indicate termination
                       NULL, NULL)) ;
        int img_id = 0;
        bool dequeued;
        do {
            dequeued = dequeue(&img_id);
        } while (!dequeued || img_id != -1);
    }
};


class server_queues_context : public rdma_server_context {
private:
    queues_gpu_context gpu_context;
    int blocks;
    /* TODO: add memory region(s) for CPU-GPU queues */
    struct ibv_mr *mr_cpu_to_gpu; 
    struct ibv_mr *mr_gpu_to_cpu; 
    queue<cpu_to_gpu_entry> *cpu_to_gpu;
    queue<gpu_to_cpu_entry> *gpu_to_cpu;
    
public:
    explicit server_queues_context(uint16_t tcp_port) : 
        rdma_server_context(tcp_port),
        gpu_context(queues_gpu_context(256)){
        
        /* TODO Initialize additional server MRs as needed. */
        blocks = gpu_context.getBlocks();
        gpu_context.getQueues(&cpu_to_gpu , &gpu_to_cpu); //get pointers to queues
        
        // register the memory regions
        mr_cpu_to_gpu = ibv_reg_mr(pd, cpu_to_gpu, sizeof(queue<cpu_to_gpu_entry>[blocks]) , IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
        if (!mr_cpu_to_gpu) {
            perror("ibv_reg_mr() failed for mr_cpu_to_gpu");
            exit(1);
        }
        mr_gpu_to_cpu = ibv_reg_mr(pd, gpu_to_cpu, sizeof(queue<gpu_to_cpu_entry>[blocks]) , IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
        if (!mr_gpu_to_cpu) {
            perror("ibv_reg_mr() failed for mr_gpu_to_cpu");
            exit(1);
        }

        struct rpc_request connectionContext[2];
        
        connectionContext[0].request_id = blocks;
        connectionContext[0].input_rkey = mr_images_in->rkey;
        connectionContext[0].input_length = OUTSTANDING_REQUESTS * IMG_SZ;
        connectionContext[0].input_addr = (uint64_t) images_in;
        connectionContext[0].output_rkey = mr_images_out->rkey;
        connectionContext[0].output_length = OUTSTANDING_REQUESTS * IMG_SZ;
        connectionContext[0].output_addr = (uint64_t) images_out;
        connectionContext[1].request_id = blocks;
        connectionContext[1].input_rkey = mr_cpu_to_gpu->rkey;
        connectionContext[1].input_length = sizeof(queue<cpu_to_gpu_entry>[blocks]);
        connectionContext[1].input_addr = (uint64_t) cpu_to_gpu;
        connectionContext[1].output_rkey = mr_gpu_to_cpu->rkey;
        connectionContext[1].output_length = sizeof(queue<gpu_to_cpu_entry>[blocks]);
        connectionContext[1].output_addr = (uint64_t) gpu_to_cpu;

        /* TODO Exchange rkeys, addresses, and necessary information (e.g.
         * number of queues) with the client */
         send_over_socket(connectionContext, 2 * sizeof(rpc_request));
        
    }

    ~server_queues_context(){
        /* TODO destroy the additional server MRs here */
        ibv_dereg_mr(mr_cpu_to_gpu);
        ibv_dereg_mr(mr_gpu_to_cpu);
    }

    virtual void event_loop() override{
        /* TODO simplified version of server_rpc_context::event_loop. As the
         * client use one sided operations, we only need one kind of message to
         * terminate the server at the end. */
        
		bool terminate = false;
		rpc_request* req;
		
        while (!terminate) {
            // Step 1: Poll for CQE
            struct ibv_wc wc;
            int ncqes = ibv_poll_cq(cq, 1, &wc);
            if (ncqes < 0) {
                perror("ibv_poll_cq() failed");
                exit(1);
            }
            if (ncqes > 0) {
		        VERBS_WC_CHECK(wc);
                switch (wc.opcode) {
                case IBV_WC_RECV:
                    /* Received a new request from the client */
                    req = &requests[0];

                    /* Terminate signal */
                    if (req->request_id == -1) {
                        printf("Terminating...\n");
                        terminate = true;
                    }
                    break;
                default:
                    printf("Unexpected completion\n");
                    assert(false);
                }
            }
		}
    }
};
    
struct queue_context{
    int pi;
    int ci;
    cpu_to_gpu_entry c2g;
    gpu_to_cpu_entry g2c;
};

class client_queues_context : public rdma_client_context {
private:
    /* TODO add necessary context to track the client side of the GPU's
     * producer/consumer queues */
	
	uint32_t requests_enqueued = 0;
    uint32_t requests_dequeued = 0;
	uchar* out_images;
    int blocks;
    int producer_nextBlockIdx = 0, consumer_nextBlockIdx = 0;
    struct queue_context q_context;
    struct rpc_request connectionContext[2];
    struct ibv_mr *mr_images_in; /* Memory region for input images */
    struct ibv_mr *mr_images_out; /* Memory region for output images */
    struct ibv_mr *mr_queue_context;
    /* TODO define other memory regions used by the client here */

public:
    client_queues_context(uint16_t tcp_port) : rdma_client_context(tcp_port)
    {
        /* TODO communicate with server to discover number of queues, necessary
         * rkeys / address, or other additional information needed to operate
         * the GPU queues remotely. */
        queue<cpu_to_gpu_entry> c2gTempQueue;
        queue<gpu_to_cpu_entry> g2cTempQueue;

        recv_over_socket(connectionContext, 2 * sizeof(rpc_request));
        blocks = connectionContext[0].request_id;
        mr_queue_context = ibv_reg_mr(pd, &q_context, sizeof(queue_context), IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
        if (!mr_queue_context) {
            perror("ibv_reg_mr() failed for input images");
            exit(1);
        }
    }

    ~client_queues_context()
    {
	/* TODO terminate the server and release memory regions and other resources */
        ibv_dereg_mr(mr_queue_context);
    }

    virtual void set_input_images(uchar *images_in, size_t bytes) override
    {
        // TODO register memory
        /* register a memory region for the input images. */
        mr_images_in = ibv_reg_mr(pd, images_in, bytes, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
        if (!mr_images_in) {
            perror("ibv_reg_mr() failed for input images");
            exit(1);
        }
    }

    virtual void set_output_images(uchar *images_out, size_t bytes) override
    {
        // TODO register memory
        /* register a memory region for the output images. */
		out_images = images_out;
        mr_images_out = ibv_reg_mr(pd, images_out, bytes, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
        if (!mr_images_out) {
            perror("ibv_reg_mr() failed for output images");
            exit(1);
        }
    }

    virtual bool enqueue(int img_id, uchar *img_in, uchar *img_out) override
    {
        /* TODO use RDMA Write and RDMA Read operations to enqueue the task on
         * a CPU-GPU producer consumer queue running on the server. */
		
		if (requests_enqueued - requests_dequeued == OUTSTANDING_REQUESTS)
            return false;

		uint32_t rkey = connectionContext[1].input_rkey;
        int num_of_used_slots = 0 , piV = 0 , ciV = 0;
		queue<cpu_to_gpu_entry>* queue_ptr = (queue<cpu_to_gpu_entry>*) (connectionContext[1].input_addr);
        for(int count = 0; count < blocks ; count++ , producer_nextBlockIdx = ((producer_nextBlockIdx + 1) % blocks) ){
            num_of_used_slots = queueStatus((uint64_t)&(queue_ptr[producer_nextBlockIdx].pi), (uint64_t)&(queue_ptr[producer_nextBlockIdx].ci), rkey , &piV , &ciV);
            if(num_of_used_slots < NSLOTS) break;
        }
        if(num_of_used_slots == NSLOTS) return false;
        write_image(&(queue_ptr[producer_nextBlockIdx]), img_id, img_in, img_out, piV);
		producer_nextBlockIdx = ((producer_nextBlockIdx + 1) % blocks);
		
		++requests_enqueued;
		std::cout << "enqueued:" << requests_enqueued << " img id:" << img_id << std::endl;
        return true;
    }

    virtual bool dequeue(int *img_id) override
    {
        /* TODO use RDMA Write and RDMA Read operations to detect the completion and dequeue a processed image
         * through a CPU-GPU producer consumer queue running on the server. */
        
        uint32_t rkey = connectionContext[1].output_rkey;
        int num_of_used_slots = 0 , piV = 0 , ciV = 0;
		queue<gpu_to_cpu_entry>* queue_ptr = (queue<gpu_to_cpu_entry>*) (connectionContext[1].output_addr);
        for(int count = 0; count < blocks ; count++ , consumer_nextBlockIdx = ((consumer_nextBlockIdx + 1) % blocks) ){
            num_of_used_slots = queueStatus((uint64_t)&(queue_ptr[consumer_nextBlockIdx].pi), (uint64_t)&(queue_ptr[consumer_nextBlockIdx].ci), rkey , &piV , &ciV);
            if(num_of_used_slots < NSLOTS) break;
        }
        if(num_of_used_slots == 0) return false;
        read_image(&(queue_ptr[consumer_nextBlockIdx]), img_id, ciV);
		consumer_nextBlockIdx = ((consumer_nextBlockIdx + 1) % blocks);
		
		++requests_dequeued;
		std::cout << "dequeued:" << requests_dequeued << " img id:" << *img_id << std::endl;
        return true;
    }
    

    int queueStatus(uint64_t pi_ptr, uint64_t ci_ptr, uint32_t rkey, int *piVal , int *ciVal){
        int pi = 0 , ci = 0;

        //rdma read pi
        post_rdma_read(
            &(q_context.pi),              // local_src
            sizeof(int),                // len
            mr_queue_context->lkey,     // lkey
            pi_ptr,                // remote_dst
            rkey,                       // rkey
            1);                         // wr_id
        
        //rdma read ci
        post_rdma_read(
            &(q_context.ci),              // local_src
            sizeof(int),                // len
            mr_queue_context->lkey,     // lkey
            ci_ptr,                // remote_dst
            rkey,                       // rkey
            2);                         // wr_id
        
        bool pi_rcv = false , ci_rcv = false;
        
        while (!pi_rcv || !ci_rcv) {
            struct ibv_wc wc;
            int ncqes = ibv_poll_cq(cq, 1, &wc);
            if (ncqes < 0) {
                perror("ibv_poll_cq() failed");
                exit(1);
            }
            if (ncqes > 0) {
                VERBS_WC_CHECK(wc);
                if (wc.opcode == IBV_WC_RDMA_READ) {
                    if(wc.wr_id == 1) {
                        pi = q_context.pi;
                        pi_rcv = true;
                    }
                    if(wc.wr_id == 2){
                         ci = q_context.ci;
                         ci_rcv = true;
                    } 
                }
                else{
                    printf("Unexpected completion\n");
                    assert(false);
                }
            }
        }
        *piVal = pi;
        *ciVal = ci;
        return pi - ci;
    }

    void write_image(queue<cpu_to_gpu_entry>* queue_ptr, int img_id, uchar *img_in, uchar *img_out , int pi){
        
		uint64_t remote_dst = connectionContext[0].input_addr + IMG_SZ * (img_id % OUTSTANDING_REQUESTS);
        uint32_t rkey = connectionContext[0].input_rkey;
        
		//Write image data
        post_rdma_write(remote_dst, IMG_SZ , rkey, img_in, mr_images_in->lkey, 0, nullptr);
        
		//post image to queue:
        //a. update data entry
        q_context.c2g.img_idx = img_id;
        q_context.c2g.img_in = (uchar*)remote_dst;
        q_context.c2g.img_out = (uchar*)(connectionContext[0].output_addr + IMG_SZ * (img_id % OUTSTANDING_REQUESTS));

        rkey = connectionContext[1].input_rkey;
        post_rdma_write((uint64_t)&(queue_ptr->data[pi % NSLOTS]), sizeof(cpu_to_gpu_entry) , rkey, &(q_context.c2g), mr_queue_context->lkey, 1, nullptr);
        
		//b. update pi
		q_context.pi = pi + 1;
        rkey = connectionContext[1].input_rkey;
		
        post_rdma_write((uint64_t)&(queue_ptr->pi), sizeof(int) , rkey, &(q_context.pi), mr_queue_context->lkey, 2, nullptr);
        
		bool imgSent = false , entrySent = false , piSent = false;
        while (!imgSent || !entrySent || !piSent) {
            // Step 1: Poll for CQE
            struct ibv_wc wc;
            int ncqes = ibv_poll_cq(cq, 1, &wc);
            if (ncqes < 0) {
                perror("ibv_poll_cq() failed");
                exit(1);
            }
            if (ncqes > 0) {
		        VERBS_WC_CHECK(wc);
                if( wc.opcode == IBV_WC_RDMA_WRITE) {    
                    if(wc.wr_id == 0) imgSent = true;
                    if(wc.wr_id == 1) entrySent = true;
                    if(wc.wr_id == 2) piSent = true;
                }
                else{
                    printf("Unexpected completion\n");
                    assert(false);
                }
            }
        }
    }
	
	void read_image(queue<gpu_to_cpu_entry>* queue_ptr, int* img_id , int ci){

		//receive image id from queue
        uint32_t rkey = connectionContext[1].output_rkey;
		post_rdma_read(&(q_context.g2c), sizeof(gpu_to_cpu_entry) , mr_queue_context->lkey, (uint64_t)&(queue_ptr->data[ci % NSLOTS]), rkey, 0);
		
		bool entryRead = false;
		while (!entryRead) {
            // Step 1: Poll for CQE
            struct ibv_wc wc;
            int ncqes = ibv_poll_cq(cq, 1, &wc);
            if (ncqes < 0) {
                perror("ibv_poll_cq() failed");
                exit(1);
            }
            if (ncqes > 0) {
		        VERBS_WC_CHECK(wc);
                if(wc.opcode == IBV_WC_RDMA_READ) {
                    if(wc.wr_id == 0) entryRead = true;
                } else {
                    printf("Unexpected completion\n");
                    assert(false);
                }
            }
        }
		
		*img_id = q_context.g2c.img_idx;

		//receive image from queue
        uint64_t remote_dst = connectionContext[0].output_addr + IMG_SZ * ((*img_id) % OUTSTANDING_REQUESTS);
        rkey = connectionContext[0].output_rkey;
        post_rdma_read(out_images + ((*img_id) % N_IMAGES) * IMG_SZ, IMG_SZ , mr_images_out->lkey, remote_dst, rkey, 1);
		
		//update ci
		q_context.ci = ci + 1;
        rkey = connectionContext[1].output_rkey;
        post_rdma_write((uint64_t)&(queue_ptr->ci), sizeof(int) , rkey, &(q_context.ci), mr_queue_context->lkey, 2);
		
        bool imgRead = false, ciSent = false;
        while (!imgRead || !ciSent) {
            // Step 1: Poll for CQE
            struct ibv_wc wc;
            int ncqes = ibv_poll_cq(cq, 1, &wc);
            if (ncqes < 0) {
                perror("ibv_poll_cq() failed");
                exit(1);
            }
            if (ncqes > 0) {
		        VERBS_WC_CHECK(wc);
                if(wc.opcode == IBV_WC_RDMA_READ) {
                    if(wc.wr_id == 1) imgRead = true;
                }
                else if(wc.opcode == IBV_WC_RDMA_WRITE) {
					if(wc.wr_id == 2) ciSent = true;
				} else {
                    printf("Unexpected completion\n");
                    assert(false);
                }
            }
        }
    }
};

std::unique_ptr<rdma_server_context> create_server(mode_enum mode, uint16_t tcp_port)
{
    switch (mode) {
    case MODE_RPC_SERVER:
        return std::make_unique<server_rpc_context>(tcp_port);
    case MODE_QUEUE:
        return std::make_unique<server_queues_context>(tcp_port);
    default:
        printf("Unknown mode.\n");
        exit(1);
    }
}

std::unique_ptr<rdma_client_context> create_client(mode_enum mode, uint16_t tcp_port)
{
    switch (mode) {
    case MODE_RPC_SERVER:
        return std::make_unique<client_rpc_context>(tcp_port);
    case MODE_QUEUE:
        return std::make_unique<client_queues_context>(tcp_port);
    default:
        printf("Unknown mode.\n");
        exit(1);
    }
}
