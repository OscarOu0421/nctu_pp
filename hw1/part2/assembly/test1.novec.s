	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 10, 15	sdk_version 10, 15, 6
	.section	__TEXT,__literal16,16byte_literals
	.p2align	4               ## -- Begin function _Z5test1PfS_S_i
LCPI0_0:
	.long	1127219200              ## 0x43300000
	.long	1160773632              ## 0x45300000
	.long	0                       ## 0x0
	.long	0                       ## 0x0
LCPI0_1:
	.quad	4841369599423283200     ## double 4503599627370496
	.quad	4985484787499139072     ## double 1.9342813113834067E+25
	.section	__TEXT,__literal8,8byte_literals
	.p2align	3
LCPI0_2:
	.quad	4472406533629990549     ## double 1.0000000000000001E-9
	.section	__TEXT,__text,regular,pure_instructions
	.globl	__Z5test1PfS_S_i
	.p2align	4, 0x90
__Z5test1PfS_S_i:                       ## @_Z5test1PfS_S_i
	.cfi_startproc
## %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	pushq	%rax
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	movq	%rdx, %r15
	movq	%rsi, %r12
	movq	%rdi, %rbx
	xorl	%r13d, %r13d
	callq	_mach_absolute_time
	movq	%rax, %r14
	.p2align	4, 0x90
LBB0_1:                                 ## =>This Loop Header: Depth=1
                                        ##     Child Loop BB0_2 Depth 2
	xorl	%eax, %eax
	.p2align	4, 0x90
LBB0_2:                                 ##   Parent Loop BB0_1 Depth=1
                                        ## =>  This Inner Loop Header: Depth=2
	movss	(%rbx,%rax,4), %xmm0    ## xmm0 = mem[0],zero,zero,zero
	addss	(%r12,%rax,4), %xmm0
	movss	%xmm0, (%r15,%rax,4)
	movss	4(%rbx,%rax,4), %xmm0   ## xmm0 = mem[0],zero,zero,zero
	addss	4(%r12,%rax,4), %xmm0
	movss	%xmm0, 4(%r15,%rax,4)
	movss	8(%rbx,%rax,4), %xmm0   ## xmm0 = mem[0],zero,zero,zero
	addss	8(%r12,%rax,4), %xmm0
	movss	%xmm0, 8(%r15,%rax,4)
	movss	12(%rbx,%rax,4), %xmm0  ## xmm0 = mem[0],zero,zero,zero
	addss	12(%r12,%rax,4), %xmm0
	movss	%xmm0, 12(%r15,%rax,4)
	addq	$4, %rax
	cmpq	$1024, %rax             ## imm = 0x400
	jne	LBB0_2
## %bb.3:                               ##   in Loop: Header=BB0_1 Depth=1
	incl	%r13d
	cmpl	$20000000, %r13d        ## imm = 0x1312D00
	jne	LBB0_1
## %bb.4:
	callq	_mach_absolute_time
	movq	%rax, %rbx
	leaq	__ZZL5tdiffyyE8timebase(%rip), %rdi
	callq	_mach_timebase_info
	testl	%eax, %eax
	jne	LBB0_6
## %bb.5:
	subq	%r14, %rbx
	movq	%rbx, %xmm0
	punpckldq	LCPI0_0(%rip), %xmm0 ## xmm0 = xmm0[0],mem[0],xmm0[1],mem[1]
	subpd	LCPI0_1(%rip), %xmm0
	movapd	%xmm0, %xmm1
	unpckhpd	%xmm0, %xmm1    ## xmm1 = xmm1[1],xmm0[1]
	movl	__ZZL5tdiffyyE8timebase(%rip), %eax
	cvtsi2sd	%rax, %xmm2
	addsd	%xmm0, %xmm1
	movl	__ZZL5tdiffyyE8timebase+4(%rip), %eax
	xorps	%xmm0, %xmm0
	cvtsi2sd	%rax, %xmm0
	mulsd	%xmm1, %xmm2
	divsd	%xmm0, %xmm2
	mulsd	LCPI0_2(%rip), %xmm2
	movsd	%xmm2, -48(%rbp)        ## 8-byte Spill
	movq	__ZNSt3__14coutE@GOTPCREL(%rip), %rdi
	leaq	L_.str(%rip), %rsi
	movl	$47, %edx
	callq	__ZNSt3__124__put_character_sequenceIcNS_11char_traitsIcEEEERNS_13basic_ostreamIT_T0_EES7_PKS4_m
	movq	%rax, %rdi
	movsd	-48(%rbp), %xmm0        ## 8-byte Reload
                                        ## xmm0 = mem[0],zero
	callq	__ZNSt3__113basic_ostreamIcNS_11char_traitsIcEEElsEd
	leaq	L_.str.1(%rip), %rsi
	movl	$8, %edx
	movq	%rax, %rdi
	callq	__ZNSt3__124__put_character_sequenceIcNS_11char_traitsIcEEEERNS_13basic_ostreamIT_T0_EES7_PKS4_m
	movq	%rax, %rdi
	movl	$1024, %esi             ## imm = 0x400
	callq	__ZNSt3__113basic_ostreamIcNS_11char_traitsIcEEElsEi
	leaq	L_.str.2(%rip), %rsi
	movl	$5, %edx
	movq	%rax, %rdi
	callq	__ZNSt3__124__put_character_sequenceIcNS_11char_traitsIcEEEERNS_13basic_ostreamIT_T0_EES7_PKS4_m
	movq	%rax, %rdi
	movl	$20000000, %esi         ## imm = 0x1312D00
	callq	__ZNSt3__113basic_ostreamIcNS_11char_traitsIcEEElsEi
	leaq	L_.str.3(%rip), %rsi
	movl	$2, %edx
	movq	%rax, %rdi
	addq	$8, %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	jmp	__ZNSt3__124__put_character_sequenceIcNS_11char_traitsIcEEEERNS_13basic_ostreamIT_T0_EES7_PKS4_m ## TAILCALL
LBB0_6:
	callq	__Z5test1PfS_S_i.cold.1
	.cfi_endproc
                                        ## -- End function
	.globl	__ZNSt3__124__put_character_sequenceIcNS_11char_traitsIcEEEERNS_13basic_ostreamIT_T0_EES7_PKS4_m ## -- Begin function _ZNSt3__124__put_character_sequenceIcNS_11char_traitsIcEEEERNS_13basic_ostreamIT_T0_EES7_PKS4_m
	.weak_def_can_be_hidden	__ZNSt3__124__put_character_sequenceIcNS_11char_traitsIcEEEERNS_13basic_ostreamIT_T0_EES7_PKS4_m
	.p2align	4, 0x90
__ZNSt3__124__put_character_sequenceIcNS_11char_traitsIcEEEERNS_13basic_ostreamIT_T0_EES7_PKS4_m: ## @_ZNSt3__124__put_character_sequenceIcNS_11char_traitsIcEEEERNS_13basic_ostreamIT_T0_EES7_PKS4_m
Lfunc_begin0:
	.cfi_startproc
	.cfi_personality 155, ___gxx_personality_v0
	.cfi_lsda 16, Lexception0
## %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	subq	$40, %rsp
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	movq	%rdx, %r14
	movq	%rsi, %r15
	movq	%rdi, %rbx
Ltmp0:
	leaq	-80(%rbp), %rdi
	movq	%rbx, %rsi
	callq	__ZNSt3__113basic_ostreamIcNS_11char_traitsIcEEE6sentryC1ERS3_
Ltmp1:
## %bb.1:
	cmpb	$0, -80(%rbp)
	je	LBB1_10
## %bb.2:
	movq	(%rbx), %rax
	movq	-24(%rax), %rax
	leaq	(%rbx,%rax), %r12
	movq	40(%rbx,%rax), %rdi
	movl	8(%rbx,%rax), %r13d
	movl	144(%rbx,%rax), %eax
	cmpl	$-1, %eax
	jne	LBB1_7
## %bb.3:
Ltmp3:
	movq	%rdi, -64(%rbp)         ## 8-byte Spill
	leaq	-56(%rbp), %rdi
	movq	%r12, %rsi
	callq	__ZNKSt3__18ios_base6getlocEv
Ltmp4:
## %bb.4:
Ltmp5:
	movq	__ZNSt3__15ctypeIcE2idE@GOTPCREL(%rip), %rsi
	leaq	-56(%rbp), %rdi
	callq	__ZNKSt3__16locale9use_facetERNS0_2idE
Ltmp6:
## %bb.5:
	movq	(%rax), %rcx
Ltmp7:
	movq	%rax, %rdi
	movl	$32, %esi
	callq	*56(%rcx)
	movb	%al, -41(%rbp)          ## 1-byte Spill
Ltmp8:
## %bb.6:
	leaq	-56(%rbp), %rdi
	callq	__ZNSt3__16localeD1Ev
	movsbl	-41(%rbp), %eax         ## 1-byte Folded Reload
	movl	%eax, 144(%r12)
	movq	-64(%rbp), %rdi         ## 8-byte Reload
LBB1_7:
	addq	%r15, %r14
	andl	$176, %r13d
	cmpl	$32, %r13d
	movq	%r15, %rdx
	cmoveq	%r14, %rdx
Ltmp10:
	movsbl	%al, %r9d
	movq	%r15, %rsi
	movq	%r14, %rcx
	movq	%r12, %r8
	callq	__ZNSt3__116__pad_and_outputIcNS_11char_traitsIcEEEENS_19ostreambuf_iteratorIT_T0_EES6_PKS4_S8_S8_RNS_8ios_baseES4_
Ltmp11:
## %bb.8:
	testq	%rax, %rax
	jne	LBB1_10
## %bb.9:
	movq	(%rbx), %rax
	movq	-24(%rax), %rax
	leaq	(%rbx,%rax), %rdi
	movl	32(%rbx,%rax), %esi
	orl	$5, %esi
Ltmp13:
	callq	__ZNSt3__18ios_base5clearEj
Ltmp14:
LBB1_10:
	leaq	-80(%rbp), %rdi
	callq	__ZNSt3__113basic_ostreamIcNS_11char_traitsIcEEE6sentryD1Ev
LBB1_11:
	movq	%rbx, %rax
	addq	$40, %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	retq
LBB1_12:
Ltmp15:
	jmp	LBB1_15
LBB1_13:
Ltmp9:
	movq	%rax, %r14
	leaq	-56(%rbp), %rdi
	callq	__ZNSt3__16localeD1Ev
	jmp	LBB1_16
LBB1_14:
Ltmp12:
LBB1_15:
	movq	%rax, %r14
LBB1_16:
	leaq	-80(%rbp), %rdi
	callq	__ZNSt3__113basic_ostreamIcNS_11char_traitsIcEEE6sentryD1Ev
	jmp	LBB1_18
LBB1_17:
Ltmp2:
	movq	%rax, %r14
LBB1_18:
	movq	%r14, %rdi
	callq	___cxa_begin_catch
	movq	(%rbx), %rax
	movq	-24(%rax), %rdi
	addq	%rbx, %rdi
Ltmp16:
	callq	__ZNSt3__18ios_base33__set_badbit_and_consider_rethrowEv
Ltmp17:
## %bb.19:
	callq	___cxa_end_catch
	jmp	LBB1_11
LBB1_20:
Ltmp18:
	movq	%rax, %rbx
Ltmp19:
	callq	___cxa_end_catch
Ltmp20:
## %bb.21:
	movq	%rbx, %rdi
	callq	__Unwind_Resume
	ud2
LBB1_22:
Ltmp21:
	movq	%rax, %rdi
	callq	___clang_call_terminate
Lfunc_end0:
	.cfi_endproc
	.section	__TEXT,__gcc_except_tab
	.p2align	2
GCC_except_table1:
Lexception0:
	.byte	255                     ## @LPStart Encoding = omit
	.byte	155                     ## @TType Encoding = indirect pcrel sdata4
	.uleb128 Lttbase0-Lttbaseref0
Lttbaseref0:
	.byte	1                       ## Call site Encoding = uleb128
	.uleb128 Lcst_end0-Lcst_begin0
Lcst_begin0:
	.uleb128 Ltmp0-Lfunc_begin0     ## >> Call Site 1 <<
	.uleb128 Ltmp1-Ltmp0            ##   Call between Ltmp0 and Ltmp1
	.uleb128 Ltmp2-Lfunc_begin0     ##     jumps to Ltmp2
	.byte	1                       ##   On action: 1
	.uleb128 Ltmp3-Lfunc_begin0     ## >> Call Site 2 <<
	.uleb128 Ltmp4-Ltmp3            ##   Call between Ltmp3 and Ltmp4
	.uleb128 Ltmp12-Lfunc_begin0    ##     jumps to Ltmp12
	.byte	1                       ##   On action: 1
	.uleb128 Ltmp5-Lfunc_begin0     ## >> Call Site 3 <<
	.uleb128 Ltmp8-Ltmp5            ##   Call between Ltmp5 and Ltmp8
	.uleb128 Ltmp9-Lfunc_begin0     ##     jumps to Ltmp9
	.byte	1                       ##   On action: 1
	.uleb128 Ltmp10-Lfunc_begin0    ## >> Call Site 4 <<
	.uleb128 Ltmp11-Ltmp10          ##   Call between Ltmp10 and Ltmp11
	.uleb128 Ltmp12-Lfunc_begin0    ##     jumps to Ltmp12
	.byte	1                       ##   On action: 1
	.uleb128 Ltmp13-Lfunc_begin0    ## >> Call Site 5 <<
	.uleb128 Ltmp14-Ltmp13          ##   Call between Ltmp13 and Ltmp14
	.uleb128 Ltmp15-Lfunc_begin0    ##     jumps to Ltmp15
	.byte	1                       ##   On action: 1
	.uleb128 Ltmp14-Lfunc_begin0    ## >> Call Site 6 <<
	.uleb128 Ltmp16-Ltmp14          ##   Call between Ltmp14 and Ltmp16
	.byte	0                       ##     has no landing pad
	.byte	0                       ##   On action: cleanup
	.uleb128 Ltmp16-Lfunc_begin0    ## >> Call Site 7 <<
	.uleb128 Ltmp17-Ltmp16          ##   Call between Ltmp16 and Ltmp17
	.uleb128 Ltmp18-Lfunc_begin0    ##     jumps to Ltmp18
	.byte	0                       ##   On action: cleanup
	.uleb128 Ltmp17-Lfunc_begin0    ## >> Call Site 8 <<
	.uleb128 Ltmp19-Ltmp17          ##   Call between Ltmp17 and Ltmp19
	.byte	0                       ##     has no landing pad
	.byte	0                       ##   On action: cleanup
	.uleb128 Ltmp19-Lfunc_begin0    ## >> Call Site 9 <<
	.uleb128 Ltmp20-Ltmp19          ##   Call between Ltmp19 and Ltmp20
	.uleb128 Ltmp21-Lfunc_begin0    ##     jumps to Ltmp21
	.byte	1                       ##   On action: 1
	.uleb128 Ltmp20-Lfunc_begin0    ## >> Call Site 10 <<
	.uleb128 Lfunc_end0-Ltmp20      ##   Call between Ltmp20 and Lfunc_end0
	.byte	0                       ##     has no landing pad
	.byte	0                       ##   On action: cleanup
Lcst_end0:
	.byte	1                       ## >> Action Record 1 <<
                                        ##   Catch TypeInfo 1
	.byte	0                       ##   No further actions
	.p2align	2
                                        ## >> Catch TypeInfos <<
	.long	0                       ## TypeInfo 1
Lttbase0:
	.p2align	2
                                        ## -- End function
	.section	__TEXT,__text,regular,pure_instructions
	.private_extern	__ZNSt3__116__pad_and_outputIcNS_11char_traitsIcEEEENS_19ostreambuf_iteratorIT_T0_EES6_PKS4_S8_S8_RNS_8ios_baseES4_ ## -- Begin function _ZNSt3__116__pad_and_outputIcNS_11char_traitsIcEEEENS_19ostreambuf_iteratorIT_T0_EES6_PKS4_S8_S8_RNS_8ios_baseES4_
	.globl	__ZNSt3__116__pad_and_outputIcNS_11char_traitsIcEEEENS_19ostreambuf_iteratorIT_T0_EES6_PKS4_S8_S8_RNS_8ios_baseES4_
	.weak_def_can_be_hidden	__ZNSt3__116__pad_and_outputIcNS_11char_traitsIcEEEENS_19ostreambuf_iteratorIT_T0_EES6_PKS4_S8_S8_RNS_8ios_baseES4_
	.p2align	4, 0x90
__ZNSt3__116__pad_and_outputIcNS_11char_traitsIcEEEENS_19ostreambuf_iteratorIT_T0_EES6_PKS4_S8_S8_RNS_8ios_baseES4_: ## @_ZNSt3__116__pad_and_outputIcNS_11char_traitsIcEEEENS_19ostreambuf_iteratorIT_T0_EES6_PKS4_S8_S8_RNS_8ios_baseES4_
Lfunc_begin1:
	.cfi_startproc
	.cfi_personality 155, ___gxx_personality_v0
	.cfi_lsda 16, Lexception1
## %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	subq	$56, %rsp
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	testq	%rdi, %rdi
	je	LBB2_20
## %bb.1:
	movq	%r8, %r12
	movq	%rcx, %r15
	movq	%rdi, %r13
	movl	%r9d, -68(%rbp)         ## 4-byte Spill
	movq	%rcx, %rax
	subq	%rsi, %rax
	movq	24(%r8), %rcx
	xorl	%r14d, %r14d
	subq	%rax, %rcx
	cmovgq	%rcx, %r14
	movq	%rdx, -88(%rbp)         ## 8-byte Spill
	movq	%rdx, %rbx
	subq	%rsi, %rbx
	testq	%rbx, %rbx
	jle	LBB2_3
## %bb.2:
	movq	(%r13), %rax
	movq	%r13, %rdi
	movq	%rbx, %rdx
	callq	*96(%rax)
	cmpq	%rbx, %rax
	jne	LBB2_20
LBB2_3:
	testq	%r14, %r14
	jle	LBB2_16
## %bb.4:
	movq	%r12, -80(%rbp)         ## 8-byte Spill
	cmpq	$23, %r14
	jae	LBB2_8
## %bb.5:
	leal	(%r14,%r14), %eax
	movb	%al, -64(%rbp)
	leaq	-64(%rbp), %rbx
	leaq	-63(%rbp), %r12
	jmp	LBB2_9
LBB2_8:
	leaq	16(%r14), %rbx
	andq	$-16, %rbx
	movq	%rbx, %rdi
	callq	__Znwm
	movq	%rax, %r12
	movq	%rax, -48(%rbp)
	orq	$1, %rbx
	movq	%rbx, -64(%rbp)
	movq	%r14, -56(%rbp)
	leaq	-64(%rbp), %rbx
LBB2_9:
	movzbl	-68(%rbp), %esi         ## 1-byte Folded Reload
	movq	%r12, %rdi
	movq	%r14, %rdx
	callq	_memset
	movb	$0, (%r12,%r14)
	testb	$1, -64(%rbp)
	je	LBB2_11
## %bb.10:
	movq	-48(%rbp), %rbx
	jmp	LBB2_12
LBB2_11:
	incq	%rbx
LBB2_12:
	movq	-80(%rbp), %r12         ## 8-byte Reload
	movq	(%r13), %rax
Ltmp22:
	movq	%r13, %rdi
	movq	%rbx, %rsi
	movq	%r14, %rdx
	callq	*96(%rax)
Ltmp23:
## %bb.13:
	movq	%rax, %rbx
	testb	$1, -64(%rbp)
	je	LBB2_15
## %bb.14:
	movq	-48(%rbp), %rdi
	callq	__ZdlPv
LBB2_15:
	cmpq	%r14, %rbx
	jne	LBB2_20
LBB2_16:
	movq	-88(%rbp), %rsi         ## 8-byte Reload
	subq	%rsi, %r15
	testq	%r15, %r15
	jle	LBB2_18
## %bb.17:
	movq	(%r13), %rax
	movq	%r13, %rdi
	movq	%r15, %rdx
	callq	*96(%rax)
	cmpq	%r15, %rax
	jne	LBB2_20
LBB2_18:
	movq	$0, 24(%r12)
	jmp	LBB2_21
LBB2_20:
	xorl	%r13d, %r13d
LBB2_21:
	movq	%r13, %rax
	addq	$56, %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	retq
LBB2_22:
Ltmp24:
	movq	%rax, %rbx
	testb	$1, -64(%rbp)
	je	LBB2_24
## %bb.23:
	movq	-48(%rbp), %rdi
	callq	__ZdlPv
LBB2_24:
	movq	%rbx, %rdi
	callq	__Unwind_Resume
	ud2
Lfunc_end1:
	.cfi_endproc
	.section	__TEXT,__gcc_except_tab
	.p2align	2
GCC_except_table2:
Lexception1:
	.byte	255                     ## @LPStart Encoding = omit
	.byte	255                     ## @TType Encoding = omit
	.byte	1                       ## Call site Encoding = uleb128
	.uleb128 Lcst_end1-Lcst_begin1
Lcst_begin1:
	.uleb128 Lfunc_begin1-Lfunc_begin1 ## >> Call Site 1 <<
	.uleb128 Ltmp22-Lfunc_begin1    ##   Call between Lfunc_begin1 and Ltmp22
	.byte	0                       ##     has no landing pad
	.byte	0                       ##   On action: cleanup
	.uleb128 Ltmp22-Lfunc_begin1    ## >> Call Site 2 <<
	.uleb128 Ltmp23-Ltmp22          ##   Call between Ltmp22 and Ltmp23
	.uleb128 Ltmp24-Lfunc_begin1    ##     jumps to Ltmp24
	.byte	0                       ##   On action: cleanup
	.uleb128 Ltmp23-Lfunc_begin1    ## >> Call Site 3 <<
	.uleb128 Lfunc_end1-Ltmp23      ##   Call between Ltmp23 and Lfunc_end1
	.byte	0                       ##     has no landing pad
	.byte	0                       ##   On action: cleanup
Lcst_end1:
	.p2align	2
                                        ## -- End function
	.section	__TEXT,__text,regular,pure_instructions
	.private_extern	___clang_call_terminate ## -- Begin function __clang_call_terminate
	.globl	___clang_call_terminate
	.weak_def_can_be_hidden	___clang_call_terminate
	.p2align	4, 0x90
___clang_call_terminate:                ## @__clang_call_terminate
## %bb.0:
	pushq	%rax
	callq	___cxa_begin_catch
	callq	__ZSt9terminatev
                                        ## -- End function
	.p2align	4, 0x90         ## -- Begin function _Z5test1PfS_S_i.cold.1
__Z5test1PfS_S_i.cold.1:                ## @_Z5test1PfS_S_i.cold.1
	.cfi_startproc
## %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	leaq	L___func__._ZL5tdiffyy(%rip), %rdi
	leaq	L_.str.4(%rip), %rsi
	leaq	L_.str.5(%rip), %rcx
	pushq	$48
	popq	%rdx
	callq	___assert_rtn
	.cfi_endproc
                                        ## -- End function
	.section	__TEXT,__cstring,cstring_literals
L_.str:                                 ## @.str
	.asciz	"Elapsed execution time of the loop in test1():\n"

L_.str.1:                               ## @.str.1
	.asciz	"sec (N: "

L_.str.2:                               ## @.str.2
	.asciz	", I: "

L_.str.3:                               ## @.str.3
	.asciz	")\n"

.zerofill __DATA,__bss,__ZZL5tdiffyyE8timebase,8,2 ## @_ZZL5tdiffyyE8timebase
L___func__._ZL5tdiffyy:                 ## @__func__._ZL5tdiffyy
	.asciz	"tdiff"

L_.str.4:                               ## @.str.4
	.asciz	"./fasttime.h"

L_.str.5:                               ## @.str.5
	.asciz	"r == 0"

.subsections_via_symbols
