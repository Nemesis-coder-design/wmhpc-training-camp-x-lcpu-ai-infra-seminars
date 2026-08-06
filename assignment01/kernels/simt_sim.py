"""问题 1.6（选做）：SIMT Simulator —— 一个 warp 的执行模拟器。

不需要 GPU

contract: 实现 run(program) -> (regs, cycles)
- warp 固定 32 个 lane，lane i 的寄存器初值为 i（int）；
- program 是指令列表，指令是元组，共三种：
    ("add", k)   active lanes 的 reg += k，1 cycle
    ("mul", k)   active lanes 的 reg *= k，1 cycle
    ("if_lt", t, then_prog, else_prog)
        reg < t 的 lane 走 then_prog，其余走 else_prog。
        模拟器先带 mask 执行 then_prog，再带 mask 的补集执行
        else_prog，然后汇合。某一支没有 active lane 时整支跳过、
        不计拍。嵌套指令照常计拍（divergence 的代价就在这里）。
        if_lt 这条指令本身不计拍，拍数只来自实际执行到的 add / mul。
- 返回值 regs 是 32 个 lane 的最终寄存器值（list），cycles 是总拍数。

通过 pytest tests/test_simt_sim.py 即为完成。
"""


class SIMTsimulator:
    def __init__(self,num_lanes=32):
        self.num_lanes=num_lanes
        self.regs=list(range(num_lanes))
        self.instructions=[]
        self.cycles=0

    def load_pg(self,instructions):
        self.instructions=instructions

    def run(self):
        self.exec(self.instructions,[True]*self.num_lanes)
        return self.regs,self.cycles

    def exec(self,instructions,mask):
        if not any(mask):
            return

        for instruction in instructions:
            op=instruction[0]

            if op=='add':
                k=instruction[1]
                for i in range(self.num_lanes):
                    if mask[i]:
                        self.regs[i]+=k
                self.cycles+=1

            elif op=='mul':
                k=instruction[1]
                for i in range(self.num_lanes):
                    if mask[i]:
                        self.regs[i]*=k
                self.cycles+=1

            elif op=='if_lt':
                t=instruction[1]
                then_prog=instruction[2]
                else_prog=instruction[3]
                true_mask=[]
                false_mask=[]

                for i in range(self.num_lanes):
                    if mask[i]:
                        if self.regs[i]<t:
                            true_mask.append(True)
                            false_mask.append(False)
                        else:
                            true_mask.append(False)
                            false_mask.append(True)
                    else:
                        true_mask.append(False)
                        false_mask.append(False)

                if any(true_mask):
                    self.exec(then_prog,true_mask)

                if any(false_mask):
                    self.exec(else_prog,false_mask)

            else:
                raise ValueError('unknown instruction: '+op)


def run(program):
    simulator=SIMTsimulator()
    simulator.load_pg(program)
    return simulator.run()

