class SIMTsimulator:
    def __init__(self,num_lanes=32):
        self.num_lanes=num_lanes
        self.regs=[[0]*8 for _ in range(num_lanes)]
        self.instuctions=[]

    def load_pg(self,instructions):
        self.instuctions=instructions

    def run(self):
        self.exec(0,[True]*self.num_lanes)

    def exec(self,pc,mask):
        while(pc<len):
            str=self.instuctions[pc]
            op=str['op']

            if op=='br':
                cond_reg=str['cond']       
                true_pc=str['true_target'] 
                false_pc=str['false_target'] 

                true_mask=[]
                false_mask=[]
                for i in range(self.num_lanes):
                    if mask[i]:  
                        if self.regs[i][cond_reg]!=0:
                            true_mask.append(True)
                            false_mask.append(False)
                        else:
                            true_mask.append(False)
                            false_mask.append(True)
                    else:        
                        true_mask.append(False)
                        false_mask.append(False)

                if any(true_mask):
                    self._exec(true_pc,true_mask)
     
                if any(false_mask):
                    self._exec(false_pc,false_mask)
                return

            elif op=='add':
                dst, src0, src1=str['dst'],str['src0'],str['src1']
                for i in range(self.num_lanes):
                    if mask[i]: 
                        self.regs[i][dst] = self.regs[i][src0]+self.regs[i][src1]
                pc+=1  

            elif op=='sub':
                dst,src0,src1=str['dst'],str['src0'],str['src1']
                for i in range(self.num_lanes):
                    if mask[i]:
                        self.regs[i][dst]=self.regs[i][src0]-self.regs[i][src1]
                pc+=1

            elif op=='mov':
                dst, src=str['dst'],str['src']
                for i in range(self.num_lanes):
                    if mask[i]:
                        self.regs[i][dst]=self.regs[i][src]
                pc+=1

            else:
                pc+=1