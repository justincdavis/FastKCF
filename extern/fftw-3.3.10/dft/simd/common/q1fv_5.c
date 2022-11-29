/*
 * Copyright (c) 2003, 2007-14 Matteo Frigo
 * Copyright (c) 2003, 2007-14 Massachusetts Institute of Technology
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 */

/* This file was automatically generated --- DO NOT EDIT */
/* Generated on Tue Sep 14 10:46:00 EDT 2021 */

#include "dft/codelet-dft.h"

#if defined(ARCH_PREFERS_FMA) || defined(ISA_EXTENSION_PREFERS_FMA)

/* Generated by: ../../../genfft/gen_twidsq_c.native -fma -simd -compact -variables 4 -pipeline-latency 8 -n 5 -dif -name q1fv_5 -include dft/simd/q1f.h */

/*
 * This function contains 100 FP additions, 95 FP multiplications,
 * (or, 55 additions, 50 multiplications, 45 fused multiply/add),
 * 44 stack variables, 4 constants, and 50 memory accesses
 */
#include "dft/simd/q1f.h"

static void q1fv_5(R *ri, R *ii, const R *W, stride rs, stride vs, INT mb, INT me, INT ms)
{
     DVK(KP559016994, +0.559016994374947424102293417182819058860154590);
     DVK(KP250000000, +0.250000000000000000000000000000000000000000000);
     DVK(KP618033988, +0.618033988749894848204586834365638117720309180);
     DVK(KP951056516, +0.951056516295153572116439333379382143405698634);
     {
	  INT m;
	  R *x;
	  x = ri;
	  for (m = mb, W = W + (mb * ((TWVL / VL) * 8)); m < me; m = m + VL, x = x + (VL * ms), W = W + (TWVL * 8), MAKE_VOLATILE_STRIDE(10, rs), MAKE_VOLATILE_STRIDE(10, vs)) {
	       V T1, Ta, Ti, Te, T8, T9, T1j, T1s, T1A, T1w, T1q, T1r, Tl, Tu, TC;
	       V Ty, Ts, Tt, TF, TO, TW, TS, TM, TN, TZ, T18, T1g, T1c, T16, T17;
	       {
		    V T7, Td, T4, Tc;
		    T1 = LD(&(x[0]), ms, &(x[0]));
		    {
			 V T5, T6, T2, T3;
			 T5 = LD(&(x[WS(rs, 2)]), ms, &(x[0]));
			 T6 = LD(&(x[WS(rs, 3)]), ms, &(x[WS(rs, 1)]));
			 T7 = VADD(T5, T6);
			 Td = VSUB(T5, T6);
			 T2 = LD(&(x[WS(rs, 1)]), ms, &(x[WS(rs, 1)]));
			 T3 = LD(&(x[WS(rs, 4)]), ms, &(x[0]));
			 T4 = VADD(T2, T3);
			 Tc = VSUB(T2, T3);
		    }
		    Ta = VSUB(T4, T7);
		    Ti = VMUL(LDK(KP951056516), VFNMS(LDK(KP618033988), Tc, Td));
		    Te = VMUL(LDK(KP951056516), VFMA(LDK(KP618033988), Td, Tc));
		    T8 = VADD(T4, T7);
		    T9 = VFNMS(LDK(KP250000000), T8, T1);
	       }
	       {
		    V T1p, T1v, T1m, T1u;
		    T1j = LD(&(x[WS(vs, 4)]), ms, &(x[WS(vs, 4)]));
		    {
			 V T1n, T1o, T1k, T1l;
			 T1n = LD(&(x[WS(vs, 4) + WS(rs, 2)]), ms, &(x[WS(vs, 4)]));
			 T1o = LD(&(x[WS(vs, 4) + WS(rs, 3)]), ms, &(x[WS(vs, 4) + WS(rs, 1)]));
			 T1p = VADD(T1n, T1o);
			 T1v = VSUB(T1n, T1o);
			 T1k = LD(&(x[WS(vs, 4) + WS(rs, 1)]), ms, &(x[WS(vs, 4) + WS(rs, 1)]));
			 T1l = LD(&(x[WS(vs, 4) + WS(rs, 4)]), ms, &(x[WS(vs, 4)]));
			 T1m = VADD(T1k, T1l);
			 T1u = VSUB(T1k, T1l);
		    }
		    T1s = VSUB(T1m, T1p);
		    T1A = VMUL(LDK(KP951056516), VFNMS(LDK(KP618033988), T1u, T1v));
		    T1w = VMUL(LDK(KP951056516), VFMA(LDK(KP618033988), T1v, T1u));
		    T1q = VADD(T1m, T1p);
		    T1r = VFNMS(LDK(KP250000000), T1q, T1j);
	       }
	       {
		    V Tr, Tx, To, Tw;
		    Tl = LD(&(x[WS(vs, 1)]), ms, &(x[WS(vs, 1)]));
		    {
			 V Tp, Tq, Tm, Tn;
			 Tp = LD(&(x[WS(vs, 1) + WS(rs, 2)]), ms, &(x[WS(vs, 1)]));
			 Tq = LD(&(x[WS(vs, 1) + WS(rs, 3)]), ms, &(x[WS(vs, 1) + WS(rs, 1)]));
			 Tr = VADD(Tp, Tq);
			 Tx = VSUB(Tp, Tq);
			 Tm = LD(&(x[WS(vs, 1) + WS(rs, 1)]), ms, &(x[WS(vs, 1) + WS(rs, 1)]));
			 Tn = LD(&(x[WS(vs, 1) + WS(rs, 4)]), ms, &(x[WS(vs, 1)]));
			 To = VADD(Tm, Tn);
			 Tw = VSUB(Tm, Tn);
		    }
		    Tu = VSUB(To, Tr);
		    TC = VMUL(LDK(KP951056516), VFNMS(LDK(KP618033988), Tw, Tx));
		    Ty = VMUL(LDK(KP951056516), VFMA(LDK(KP618033988), Tx, Tw));
		    Ts = VADD(To, Tr);
		    Tt = VFNMS(LDK(KP250000000), Ts, Tl);
	       }
	       {
		    V TL, TR, TI, TQ;
		    TF = LD(&(x[WS(vs, 2)]), ms, &(x[WS(vs, 2)]));
		    {
			 V TJ, TK, TG, TH;
			 TJ = LD(&(x[WS(vs, 2) + WS(rs, 2)]), ms, &(x[WS(vs, 2)]));
			 TK = LD(&(x[WS(vs, 2) + WS(rs, 3)]), ms, &(x[WS(vs, 2) + WS(rs, 1)]));
			 TL = VADD(TJ, TK);
			 TR = VSUB(TJ, TK);
			 TG = LD(&(x[WS(vs, 2) + WS(rs, 1)]), ms, &(x[WS(vs, 2) + WS(rs, 1)]));
			 TH = LD(&(x[WS(vs, 2) + WS(rs, 4)]), ms, &(x[WS(vs, 2)]));
			 TI = VADD(TG, TH);
			 TQ = VSUB(TG, TH);
		    }
		    TO = VSUB(TI, TL);
		    TW = VMUL(LDK(KP951056516), VFNMS(LDK(KP618033988), TQ, TR));
		    TS = VMUL(LDK(KP951056516), VFMA(LDK(KP618033988), TR, TQ));
		    TM = VADD(TI, TL);
		    TN = VFNMS(LDK(KP250000000), TM, TF);
	       }
	       {
		    V T15, T1b, T12, T1a;
		    TZ = LD(&(x[WS(vs, 3)]), ms, &(x[WS(vs, 3)]));
		    {
			 V T13, T14, T10, T11;
			 T13 = LD(&(x[WS(vs, 3) + WS(rs, 2)]), ms, &(x[WS(vs, 3)]));
			 T14 = LD(&(x[WS(vs, 3) + WS(rs, 3)]), ms, &(x[WS(vs, 3) + WS(rs, 1)]));
			 T15 = VADD(T13, T14);
			 T1b = VSUB(T13, T14);
			 T10 = LD(&(x[WS(vs, 3) + WS(rs, 1)]), ms, &(x[WS(vs, 3) + WS(rs, 1)]));
			 T11 = LD(&(x[WS(vs, 3) + WS(rs, 4)]), ms, &(x[WS(vs, 3)]));
			 T12 = VADD(T10, T11);
			 T1a = VSUB(T10, T11);
		    }
		    T18 = VSUB(T12, T15);
		    T1g = VMUL(LDK(KP951056516), VFNMS(LDK(KP618033988), T1a, T1b));
		    T1c = VMUL(LDK(KP951056516), VFMA(LDK(KP618033988), T1b, T1a));
		    T16 = VADD(T12, T15);
		    T17 = VFNMS(LDK(KP250000000), T16, TZ);
	       }
	       ST(&(x[0]), VADD(T1, T8), ms, &(x[0]));
	       ST(&(x[WS(rs, 4)]), VADD(T1j, T1q), ms, &(x[0]));
	       ST(&(x[WS(rs, 2)]), VADD(TF, TM), ms, &(x[0]));
	       ST(&(x[WS(rs, 3)]), VADD(TZ, T16), ms, &(x[WS(rs, 1)]));
	       ST(&(x[WS(rs, 1)]), VADD(Tl, Ts), ms, &(x[WS(rs, 1)]));
	       {
		    V Tj, Tk, Th, T1B, T1C, T1z;
		    Th = VFNMS(LDK(KP559016994), Ta, T9);
		    Tj = BYTWJ(&(W[TWVL * 2]), VFMAI(Ti, Th));
		    Tk = BYTWJ(&(W[TWVL * 4]), VFNMSI(Ti, Th));
		    ST(&(x[WS(vs, 2)]), Tj, ms, &(x[WS(vs, 2)]));
		    ST(&(x[WS(vs, 3)]), Tk, ms, &(x[WS(vs, 3)]));
		    T1z = VFNMS(LDK(KP559016994), T1s, T1r);
		    T1B = BYTWJ(&(W[TWVL * 2]), VFMAI(T1A, T1z));
		    T1C = BYTWJ(&(W[TWVL * 4]), VFNMSI(T1A, T1z));
		    ST(&(x[WS(vs, 2) + WS(rs, 4)]), T1B, ms, &(x[WS(vs, 2)]));
		    ST(&(x[WS(vs, 3) + WS(rs, 4)]), T1C, ms, &(x[WS(vs, 3)]));
	       }
	       {
		    V T1h, T1i, T1f, TD, TE, TB;
		    T1f = VFNMS(LDK(KP559016994), T18, T17);
		    T1h = BYTWJ(&(W[TWVL * 2]), VFMAI(T1g, T1f));
		    T1i = BYTWJ(&(W[TWVL * 4]), VFNMSI(T1g, T1f));
		    ST(&(x[WS(vs, 2) + WS(rs, 3)]), T1h, ms, &(x[WS(vs, 2) + WS(rs, 1)]));
		    ST(&(x[WS(vs, 3) + WS(rs, 3)]), T1i, ms, &(x[WS(vs, 3) + WS(rs, 1)]));
		    TB = VFNMS(LDK(KP559016994), Tu, Tt);
		    TD = BYTWJ(&(W[TWVL * 2]), VFMAI(TC, TB));
		    TE = BYTWJ(&(W[TWVL * 4]), VFNMSI(TC, TB));
		    ST(&(x[WS(vs, 2) + WS(rs, 1)]), TD, ms, &(x[WS(vs, 2) + WS(rs, 1)]));
		    ST(&(x[WS(vs, 3) + WS(rs, 1)]), TE, ms, &(x[WS(vs, 3) + WS(rs, 1)]));
	       }
	       {
		    V TX, TY, TV, TT, TU, TP;
		    TV = VFNMS(LDK(KP559016994), TO, TN);
		    TX = BYTWJ(&(W[TWVL * 2]), VFMAI(TW, TV));
		    TY = BYTWJ(&(W[TWVL * 4]), VFNMSI(TW, TV));
		    ST(&(x[WS(vs, 2) + WS(rs, 2)]), TX, ms, &(x[WS(vs, 2)]));
		    ST(&(x[WS(vs, 3) + WS(rs, 2)]), TY, ms, &(x[WS(vs, 3)]));
		    TP = VFMA(LDK(KP559016994), TO, TN);
		    TT = BYTWJ(&(W[0]), VFNMSI(TS, TP));
		    TU = BYTWJ(&(W[TWVL * 6]), VFMAI(TS, TP));
		    ST(&(x[WS(vs, 1) + WS(rs, 2)]), TT, ms, &(x[WS(vs, 1)]));
		    ST(&(x[WS(vs, 4) + WS(rs, 2)]), TU, ms, &(x[WS(vs, 4)]));
	       }
	       {
		    V Tf, Tg, Tb, Tz, TA, Tv;
		    Tb = VFMA(LDK(KP559016994), Ta, T9);
		    Tf = BYTWJ(&(W[0]), VFNMSI(Te, Tb));
		    Tg = BYTWJ(&(W[TWVL * 6]), VFMAI(Te, Tb));
		    ST(&(x[WS(vs, 1)]), Tf, ms, &(x[WS(vs, 1)]));
		    ST(&(x[WS(vs, 4)]), Tg, ms, &(x[WS(vs, 4)]));
		    Tv = VFMA(LDK(KP559016994), Tu, Tt);
		    Tz = BYTWJ(&(W[0]), VFNMSI(Ty, Tv));
		    TA = BYTWJ(&(W[TWVL * 6]), VFMAI(Ty, Tv));
		    ST(&(x[WS(vs, 1) + WS(rs, 1)]), Tz, ms, &(x[WS(vs, 1) + WS(rs, 1)]));
		    ST(&(x[WS(vs, 4) + WS(rs, 1)]), TA, ms, &(x[WS(vs, 4) + WS(rs, 1)]));
	       }
	       {
		    V T1d, T1e, T19, T1x, T1y, T1t;
		    T19 = VFMA(LDK(KP559016994), T18, T17);
		    T1d = BYTWJ(&(W[0]), VFNMSI(T1c, T19));
		    T1e = BYTWJ(&(W[TWVL * 6]), VFMAI(T1c, T19));
		    ST(&(x[WS(vs, 1) + WS(rs, 3)]), T1d, ms, &(x[WS(vs, 1) + WS(rs, 1)]));
		    ST(&(x[WS(vs, 4) + WS(rs, 3)]), T1e, ms, &(x[WS(vs, 4) + WS(rs, 1)]));
		    T1t = VFMA(LDK(KP559016994), T1s, T1r);
		    T1x = BYTWJ(&(W[0]), VFNMSI(T1w, T1t));
		    T1y = BYTWJ(&(W[TWVL * 6]), VFMAI(T1w, T1t));
		    ST(&(x[WS(vs, 1) + WS(rs, 4)]), T1x, ms, &(x[WS(vs, 1)]));
		    ST(&(x[WS(vs, 4) + WS(rs, 4)]), T1y, ms, &(x[WS(vs, 4)]));
	       }
	  }
     }
     VLEAVE();
}

static const tw_instr twinstr[] = {
     VTW(0, 1),
     VTW(0, 2),
     VTW(0, 3),
     VTW(0, 4),
     { TW_NEXT, VL, 0 }
};

static const ct_desc desc = { 5, XSIMD_STRING("q1fv_5"), twinstr, &GENUS, { 55, 50, 45, 0 }, 0, 0, 0 };

void XSIMD(codelet_q1fv_5) (planner *p) {
     X(kdft_difsq_register) (p, q1fv_5, &desc);
}
#else

/* Generated by: ../../../genfft/gen_twidsq_c.native -simd -compact -variables 4 -pipeline-latency 8 -n 5 -dif -name q1fv_5 -include dft/simd/q1f.h */

/*
 * This function contains 100 FP additions, 70 FP multiplications,
 * (or, 85 additions, 55 multiplications, 15 fused multiply/add),
 * 44 stack variables, 4 constants, and 50 memory accesses
 */
#include "dft/simd/q1f.h"

static void q1fv_5(R *ri, R *ii, const R *W, stride rs, stride vs, INT mb, INT me, INT ms)
{
     DVK(KP250000000, +0.250000000000000000000000000000000000000000000);
     DVK(KP587785252, +0.587785252292473129168705954639072768597652438);
     DVK(KP951056516, +0.951056516295153572116439333379382143405698634);
     DVK(KP559016994, +0.559016994374947424102293417182819058860154590);
     {
	  INT m;
	  R *x;
	  x = ri;
	  for (m = mb, W = W + (mb * ((TWVL / VL) * 8)); m < me; m = m + VL, x = x + (VL * ms), W = W + (TWVL * 8), MAKE_VOLATILE_STRIDE(10, rs), MAKE_VOLATILE_STRIDE(10, vs)) {
	       V T8, T7, Th, Te, T9, Ta, T1q, T1p, T1z, T1w, T1r, T1s, Ts, Tr, TB;
	       V Ty, Tt, Tu, TM, TL, TV, TS, TN, TO, T16, T15, T1f, T1c, T17, T18;
	       {
		    V T6, Td, T3, Tc;
		    T8 = LD(&(x[0]), ms, &(x[0]));
		    {
			 V T4, T5, T1, T2;
			 T4 = LD(&(x[WS(rs, 2)]), ms, &(x[0]));
			 T5 = LD(&(x[WS(rs, 3)]), ms, &(x[WS(rs, 1)]));
			 T6 = VADD(T4, T5);
			 Td = VSUB(T4, T5);
			 T1 = LD(&(x[WS(rs, 1)]), ms, &(x[WS(rs, 1)]));
			 T2 = LD(&(x[WS(rs, 4)]), ms, &(x[0]));
			 T3 = VADD(T1, T2);
			 Tc = VSUB(T1, T2);
		    }
		    T7 = VMUL(LDK(KP559016994), VSUB(T3, T6));
		    Th = VBYI(VFNMS(LDK(KP587785252), Tc, VMUL(LDK(KP951056516), Td)));
		    Te = VBYI(VFMA(LDK(KP951056516), Tc, VMUL(LDK(KP587785252), Td)));
		    T9 = VADD(T3, T6);
		    Ta = VFNMS(LDK(KP250000000), T9, T8);
	       }
	       {
		    V T1o, T1v, T1l, T1u;
		    T1q = LD(&(x[WS(vs, 4)]), ms, &(x[WS(vs, 4)]));
		    {
			 V T1m, T1n, T1j, T1k;
			 T1m = LD(&(x[WS(vs, 4) + WS(rs, 2)]), ms, &(x[WS(vs, 4)]));
			 T1n = LD(&(x[WS(vs, 4) + WS(rs, 3)]), ms, &(x[WS(vs, 4) + WS(rs, 1)]));
			 T1o = VADD(T1m, T1n);
			 T1v = VSUB(T1m, T1n);
			 T1j = LD(&(x[WS(vs, 4) + WS(rs, 1)]), ms, &(x[WS(vs, 4) + WS(rs, 1)]));
			 T1k = LD(&(x[WS(vs, 4) + WS(rs, 4)]), ms, &(x[WS(vs, 4)]));
			 T1l = VADD(T1j, T1k);
			 T1u = VSUB(T1j, T1k);
		    }
		    T1p = VMUL(LDK(KP559016994), VSUB(T1l, T1o));
		    T1z = VBYI(VFNMS(LDK(KP587785252), T1u, VMUL(LDK(KP951056516), T1v)));
		    T1w = VBYI(VFMA(LDK(KP951056516), T1u, VMUL(LDK(KP587785252), T1v)));
		    T1r = VADD(T1l, T1o);
		    T1s = VFNMS(LDK(KP250000000), T1r, T1q);
	       }
	       {
		    V Tq, Tx, Tn, Tw;
		    Ts = LD(&(x[WS(vs, 1)]), ms, &(x[WS(vs, 1)]));
		    {
			 V To, Tp, Tl, Tm;
			 To = LD(&(x[WS(vs, 1) + WS(rs, 2)]), ms, &(x[WS(vs, 1)]));
			 Tp = LD(&(x[WS(vs, 1) + WS(rs, 3)]), ms, &(x[WS(vs, 1) + WS(rs, 1)]));
			 Tq = VADD(To, Tp);
			 Tx = VSUB(To, Tp);
			 Tl = LD(&(x[WS(vs, 1) + WS(rs, 1)]), ms, &(x[WS(vs, 1) + WS(rs, 1)]));
			 Tm = LD(&(x[WS(vs, 1) + WS(rs, 4)]), ms, &(x[WS(vs, 1)]));
			 Tn = VADD(Tl, Tm);
			 Tw = VSUB(Tl, Tm);
		    }
		    Tr = VMUL(LDK(KP559016994), VSUB(Tn, Tq));
		    TB = VBYI(VFNMS(LDK(KP587785252), Tw, VMUL(LDK(KP951056516), Tx)));
		    Ty = VBYI(VFMA(LDK(KP951056516), Tw, VMUL(LDK(KP587785252), Tx)));
		    Tt = VADD(Tn, Tq);
		    Tu = VFNMS(LDK(KP250000000), Tt, Ts);
	       }
	       {
		    V TK, TR, TH, TQ;
		    TM = LD(&(x[WS(vs, 2)]), ms, &(x[WS(vs, 2)]));
		    {
			 V TI, TJ, TF, TG;
			 TI = LD(&(x[WS(vs, 2) + WS(rs, 2)]), ms, &(x[WS(vs, 2)]));
			 TJ = LD(&(x[WS(vs, 2) + WS(rs, 3)]), ms, &(x[WS(vs, 2) + WS(rs, 1)]));
			 TK = VADD(TI, TJ);
			 TR = VSUB(TI, TJ);
			 TF = LD(&(x[WS(vs, 2) + WS(rs, 1)]), ms, &(x[WS(vs, 2) + WS(rs, 1)]));
			 TG = LD(&(x[WS(vs, 2) + WS(rs, 4)]), ms, &(x[WS(vs, 2)]));
			 TH = VADD(TF, TG);
			 TQ = VSUB(TF, TG);
		    }
		    TL = VMUL(LDK(KP559016994), VSUB(TH, TK));
		    TV = VBYI(VFNMS(LDK(KP587785252), TQ, VMUL(LDK(KP951056516), TR)));
		    TS = VBYI(VFMA(LDK(KP951056516), TQ, VMUL(LDK(KP587785252), TR)));
		    TN = VADD(TH, TK);
		    TO = VFNMS(LDK(KP250000000), TN, TM);
	       }
	       {
		    V T14, T1b, T11, T1a;
		    T16 = LD(&(x[WS(vs, 3)]), ms, &(x[WS(vs, 3)]));
		    {
			 V T12, T13, TZ, T10;
			 T12 = LD(&(x[WS(vs, 3) + WS(rs, 2)]), ms, &(x[WS(vs, 3)]));
			 T13 = LD(&(x[WS(vs, 3) + WS(rs, 3)]), ms, &(x[WS(vs, 3) + WS(rs, 1)]));
			 T14 = VADD(T12, T13);
			 T1b = VSUB(T12, T13);
			 TZ = LD(&(x[WS(vs, 3) + WS(rs, 1)]), ms, &(x[WS(vs, 3) + WS(rs, 1)]));
			 T10 = LD(&(x[WS(vs, 3) + WS(rs, 4)]), ms, &(x[WS(vs, 3)]));
			 T11 = VADD(TZ, T10);
			 T1a = VSUB(TZ, T10);
		    }
		    T15 = VMUL(LDK(KP559016994), VSUB(T11, T14));
		    T1f = VBYI(VFNMS(LDK(KP587785252), T1a, VMUL(LDK(KP951056516), T1b)));
		    T1c = VBYI(VFMA(LDK(KP951056516), T1a, VMUL(LDK(KP587785252), T1b)));
		    T17 = VADD(T11, T14);
		    T18 = VFNMS(LDK(KP250000000), T17, T16);
	       }
	       ST(&(x[0]), VADD(T8, T9), ms, &(x[0]));
	       ST(&(x[WS(rs, 4)]), VADD(T1q, T1r), ms, &(x[0]));
	       ST(&(x[WS(rs, 2)]), VADD(TM, TN), ms, &(x[0]));
	       ST(&(x[WS(rs, 3)]), VADD(T16, T17), ms, &(x[WS(rs, 1)]));
	       ST(&(x[WS(rs, 1)]), VADD(Ts, Tt), ms, &(x[WS(rs, 1)]));
	       {
		    V Tj, Tk, Ti, T1B, T1C, T1A;
		    Ti = VSUB(Ta, T7);
		    Tj = BYTWJ(&(W[TWVL * 2]), VADD(Th, Ti));
		    Tk = BYTWJ(&(W[TWVL * 4]), VSUB(Ti, Th));
		    ST(&(x[WS(vs, 2)]), Tj, ms, &(x[WS(vs, 2)]));
		    ST(&(x[WS(vs, 3)]), Tk, ms, &(x[WS(vs, 3)]));
		    T1A = VSUB(T1s, T1p);
		    T1B = BYTWJ(&(W[TWVL * 2]), VADD(T1z, T1A));
		    T1C = BYTWJ(&(W[TWVL * 4]), VSUB(T1A, T1z));
		    ST(&(x[WS(vs, 2) + WS(rs, 4)]), T1B, ms, &(x[WS(vs, 2)]));
		    ST(&(x[WS(vs, 3) + WS(rs, 4)]), T1C, ms, &(x[WS(vs, 3)]));
	       }
	       {
		    V T1h, T1i, T1g, TD, TE, TC;
		    T1g = VSUB(T18, T15);
		    T1h = BYTWJ(&(W[TWVL * 2]), VADD(T1f, T1g));
		    T1i = BYTWJ(&(W[TWVL * 4]), VSUB(T1g, T1f));
		    ST(&(x[WS(vs, 2) + WS(rs, 3)]), T1h, ms, &(x[WS(vs, 2) + WS(rs, 1)]));
		    ST(&(x[WS(vs, 3) + WS(rs, 3)]), T1i, ms, &(x[WS(vs, 3) + WS(rs, 1)]));
		    TC = VSUB(Tu, Tr);
		    TD = BYTWJ(&(W[TWVL * 2]), VADD(TB, TC));
		    TE = BYTWJ(&(W[TWVL * 4]), VSUB(TC, TB));
		    ST(&(x[WS(vs, 2) + WS(rs, 1)]), TD, ms, &(x[WS(vs, 2) + WS(rs, 1)]));
		    ST(&(x[WS(vs, 3) + WS(rs, 1)]), TE, ms, &(x[WS(vs, 3) + WS(rs, 1)]));
	       }
	       {
		    V TX, TY, TW, TT, TU, TP;
		    TW = VSUB(TO, TL);
		    TX = BYTWJ(&(W[TWVL * 2]), VADD(TV, TW));
		    TY = BYTWJ(&(W[TWVL * 4]), VSUB(TW, TV));
		    ST(&(x[WS(vs, 2) + WS(rs, 2)]), TX, ms, &(x[WS(vs, 2)]));
		    ST(&(x[WS(vs, 3) + WS(rs, 2)]), TY, ms, &(x[WS(vs, 3)]));
		    TP = VADD(TL, TO);
		    TT = BYTWJ(&(W[0]), VSUB(TP, TS));
		    TU = BYTWJ(&(W[TWVL * 6]), VADD(TS, TP));
		    ST(&(x[WS(vs, 1) + WS(rs, 2)]), TT, ms, &(x[WS(vs, 1)]));
		    ST(&(x[WS(vs, 4) + WS(rs, 2)]), TU, ms, &(x[WS(vs, 4)]));
	       }
	       {
		    V Tf, Tg, Tb, Tz, TA, Tv;
		    Tb = VADD(T7, Ta);
		    Tf = BYTWJ(&(W[0]), VSUB(Tb, Te));
		    Tg = BYTWJ(&(W[TWVL * 6]), VADD(Te, Tb));
		    ST(&(x[WS(vs, 1)]), Tf, ms, &(x[WS(vs, 1)]));
		    ST(&(x[WS(vs, 4)]), Tg, ms, &(x[WS(vs, 4)]));
		    Tv = VADD(Tr, Tu);
		    Tz = BYTWJ(&(W[0]), VSUB(Tv, Ty));
		    TA = BYTWJ(&(W[TWVL * 6]), VADD(Ty, Tv));
		    ST(&(x[WS(vs, 1) + WS(rs, 1)]), Tz, ms, &(x[WS(vs, 1) + WS(rs, 1)]));
		    ST(&(x[WS(vs, 4) + WS(rs, 1)]), TA, ms, &(x[WS(vs, 4) + WS(rs, 1)]));
	       }
	       {
		    V T1d, T1e, T19, T1x, T1y, T1t;
		    T19 = VADD(T15, T18);
		    T1d = BYTWJ(&(W[0]), VSUB(T19, T1c));
		    T1e = BYTWJ(&(W[TWVL * 6]), VADD(T1c, T19));
		    ST(&(x[WS(vs, 1) + WS(rs, 3)]), T1d, ms, &(x[WS(vs, 1) + WS(rs, 1)]));
		    ST(&(x[WS(vs, 4) + WS(rs, 3)]), T1e, ms, &(x[WS(vs, 4) + WS(rs, 1)]));
		    T1t = VADD(T1p, T1s);
		    T1x = BYTWJ(&(W[0]), VSUB(T1t, T1w));
		    T1y = BYTWJ(&(W[TWVL * 6]), VADD(T1w, T1t));
		    ST(&(x[WS(vs, 1) + WS(rs, 4)]), T1x, ms, &(x[WS(vs, 1)]));
		    ST(&(x[WS(vs, 4) + WS(rs, 4)]), T1y, ms, &(x[WS(vs, 4)]));
	       }
	  }
     }
     VLEAVE();
}

static const tw_instr twinstr[] = {
     VTW(0, 1),
     VTW(0, 2),
     VTW(0, 3),
     VTW(0, 4),
     { TW_NEXT, VL, 0 }
};

static const ct_desc desc = { 5, XSIMD_STRING("q1fv_5"), twinstr, &GENUS, { 85, 55, 15, 0 }, 0, 0, 0 };

void XSIMD(codelet_q1fv_5) (planner *p) {
     X(kdft_difsq_register) (p, q1fv_5, &desc);
}
#endif
