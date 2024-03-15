import React, { Component } from 'react';
import Swal from 'sweetalert2';
import APIServices from '../../API/apiservices';
import eventApi from '../../API/eventApi';
import Loader from '../../Component/Loader';
import DataTableComponent from '../../Component/DataTableComponent';
import TotalRow from '../../Component/TotalRow';
import { string } from '../../Constants/string';
import validator from '../../Constants/validator';
import color from '../../Constants/color'
import cookieStorage from '../../Constants/cookie-storage';
import $ from 'jquery';
import '../../App';
import './RPS.scss';
import { tree } from 'd3';
import TopMenuBar from '../../Component/TopMenuBar';

const apiServices = new APIServices();

let bcData = [];

class RPSSettings extends Component {
  constructor(props) {
    super(props);
    this.userDetails = JSON.parse(cookieStorage.getCookie('userDetails'))
    this.state = {
      monthData: [],
      monthTotalData: [],
      monthcolumns: [],
      modalCompartmentColumn: [],
      modalCompartmentData: [],
      modalData: [],
      regioncolumn: [],
      regionData: [],
      modelRegionDatas: [],
      modelregioncolumn: [],
      tableDatas: true,
      gettingMonth: window.monthNumToName(new Date().getMonth() + 1),
      gettingRegion: 'Null',
      monthTableTitle: 'NETWORK',
      tabLevel: 'Null',
      cabinOption: [],
      getCabinValue: 'Null',
      regionId: 'Null',
      countryId: 'Null',
      cityId: 'Null',
      commonOD: 'Null',
      type: 'Null',
      baseAccess: '',
      regions: [],
      regionSelected: 'Null',
      regionSelectedID: 'Null',
      countries: [],
      countrySelected: 'Null',
      countrySelectedID: 'Null',
      cities: [],
      citySelected: 'Null',
      selectedData: 'Null',
      loading: false,
      loading2: false,
      selectedRowClassRule: {
        'ag-row-disableChkbox': `${this.userDetails.id === 1 ? 'false' : 'true'}`,
      }
    }
    this.sendEvent('1', 'viewed Pos Page', 'pos', 'Pos Page');
  }

  sendEvent = (id, description, path, page_name) => {
    var eventData = {
      event_id: `${id}`,
      description: `User ${description}`,
      where_path: `/${path}`,
      page_name: `${page_name} Page`
    }
    // eventApi.sendEvent(eventData)
  }

  componentDidMount() {
    var self = this;
    let { gettingMonth, regionId, countryId, cityId, commonOD, getCabinValue, type } = this.state;

  }


  monthWiseCellClick = (params) => {
    var monththis = this;
    monththis.sendEvent('2', 'clicked on Months row', 'pos', 'Pos Page');
    let { gettingMonth, regionId, countryId, cityId, commonOD, getCabinValue, type } = this.state;
    var column = params.colDef.field;
    monththis.setState({ gettingMonth: params.data.Month })
    if (column === '') {
      params.event.stopPropagation();
      // apiServices.getPOSRegionTables(gettingMonth, getCabinValue).then(function (result) {
      //   var columnName = result[0].columnName;
      //   var posRegionata = result[0].posregiontableDatas;
      //   self.setState({ chartVisible: true, modelRegionDatas: posRegionata, modelregioncolumn: columnName })
      // });
      monththis.setState({ chartVisible: true })
    } else if (column === 'CY_B') {
      params.event.stopPropagation();
      monththis.showLoader();
      apiServices.getPOSCabinDetails(params.data.Month, regionId, countryId, cityId, commonOD, getCabinValue).then(function (result) {
        monththis.hideLoader();
        if (result) {
          var columnName = result[0].columnName;
          var cabinData = result[0].cabinData;
          var data = result[0].data;
          monththis.setState({ modalData: data, tableModalVisible: true, modalCompartmentData: cabinData, modalCompartmentColumn: columnName })
        }
      });

    } else if (column === 'CY_AL') {

      this.props.history.push('/topMarkets')

    } else {
      monththis.setState({ loading2: true })
      if (type === 'OD') {
        apiServices.getPOSDrillDownData(this.state.toggle, params.data.Month, regionId, countryId, cityId, commonOD, getCabinValue, 'OD').then((result) => {
          monththis.setState({ loading2: false })
          if (result) {
            monththis.setState({
              drillDownTotalData: result[0].totalData,
              posODFlowData: result[0].posregiontableDatas,
              posODFlowColumn: result[0].columnName,
              regionId: result[0].currentAccess.regionId,
              countryId: result[0].currentAccess.countryId,
              cityId: result[0].currentAccess.cityId,
              commonOD: result[0].currentAccess.commonOD,
            })
          }
        });
      } else if (type === 'Agency') {
        apiServices.getPOSDrillDownData(this.state.toggle, params.data.Month, regionId, countryId, cityId, commonOD, getCabinValue, 'Agency').then((result) => {
          monththis.setState({ loading2: false })
          if (result) {
            monththis.setState({
              drillDownTotalData: result[0].totalData,
              posAgentFlowDatas: result[0].posregiontableDatas,
              posAgentFlowcolumn: result[0].columnName,
              regionId: result[0].currentAccess.regionId,
              countryId: result[0].currentAccess.countryId,
              cityId: result[0].currentAccess.cityId,
              commonOD: result[0].currentAccess.commonOD,
            })
          }
        });
      } else if (type === 'Compartment') {
        apiServices.getPOSDrillDownData(this.state.toggle, params.data.Month, regionId, countryId, cityId, commonOD, getCabinValue, 'Compartment').then((result) => {
          monththis.setState({ loading2: false })
          if (result) {
            monththis.setState({
              drillDownTotalData: result[0].totalData,
              compartmentData: result[0].posregiontableDatas,
              compartmentColumn: result[0].columnName,
              regionId: result[0].currentAccess.regionId,
              countryId: result[0].currentAccess.countryId,
              cityId: result[0].currentAccess.cityId,
              commonOD: result[0].currentAccess.commonOD,
            })
          }
        });
      } else if (type === 'Ancillary') {
        apiServices.getPOSDrillDownData(this.state.toggle, params.data.Month, regionId, countryId, cityId, commonOD, getCabinValue, 'Ancillary').then((result) => {
          monththis.setState({ loading2: false })
          if (result) {
            monththis.setState({
              drillDownTotalData: result[0].totalData,
              ancillaryData: result[0].posregiontableDatas,
              ancillaryColumn: result[0].columnName,
              regionId: result[0].currentAccess.regionId,
              countryId: result[0].currentAccess.countryId,
              cityId: result[0].currentAccess.cityId,
              commonOD: result[0].currentAccess.commonOD,
            })
          }
        });
      } else {
        apiServices.getPOSDrillDownData(this.state.toggle, params.data.Month, regionId, countryId, cityId, commonOD, getCabinValue, type).then((result) => {
          monththis.setState({ loading2: false })
          if (result) {
            monththis.setState({
              drillDownTotalData: result[0].totalData,
              regionData: result[0].posregiontableDatas,
              regioncolumn: result[0].columnName,
              tabName: result[0].tabName,
              regionId: result[0].currentAccess.regionId,
              countryId: result[0].currentAccess.countryId,
              cityId: result[0].currentAccess.cityId,
              commonOD: result[0].currentAccess.commonOD,
            })
          }
        });
      }
    }
  }


  showLoader = () => {
    $("#loaderImage").addClass("loader-visible")
  }

  hideLoader = () => {
    $("#loaderImage").removeClass("loader-visible")
    $(".x_panel").addClass("opacity-fade");
    $(".top-buttons").addClass("opacity-fade");
  }

  render() {
    const { regions, countries, cities, cabinOption } = this.state;
    return (
      <div className='rps'>
        <TopMenuBar {...this.props} />
        <Loader />
        <div className="row">
          <div className="col-md-12 col-sm-12 col-xs-12 top">
            <div className="navdesign" style={{ marginTop: '0px' }}>
              <div className="col-md-4 col-sm-4 col-xs-4 toggle1">
                <h2>RPS Settings</h2>
                <section>
                  <nav>
                    <ol className="cd-breadcrumb">
                      <li onClick={this.homeHandleClick} > {this.state.baseAccess} </li>
                      {bcData.map((item) =>
                        <li onClick={this.listHandleClick} data-value={item.key} id={item.val} title={item.title}> {` > ${item.val}`} </li>
                      )}
                    </ol>
                  </nav>
                </section>
              </div>
            </div>
          </div>
        </div>

        <div className="row">
          <div className="col-md-12 col-sm-12 col-xs-12">
            <div className="x_panel">
              <div className="x_content">

                <div className="col-md-8 col-sm-8 col-xs-8">
                  <div className="form-group versions">
                    <h4>Versions:</h4>
                    <select className="form-control cabinselect"
                      onChange={(e) => { }} id="region">
                      <option selected={true} value="Version 1">Version 1</option>
                      <option value="Version 2">Version 2</option>
                      <option value="Version 3">Version 3</option>
                    </select>
                  </div>
                </div>

                <div className="col-md-8 col-sm-8 col-xs-8">
                  <div className="form-group versions">
                    <h4>Freeze Fine Tunning:</h4>
                    <input type="radio" id='freeze' name="gender" value="freeze" />
                    <label for="freeze">Freeze</label>
                    <input type="radio" id='unfreeze' name="gender" value="unfreeze" />
                    <label for="unfreeze">Un Freeze</label>
                  </div>
                </div>

                <div className="btn-main" style={{ justifyContent: 'flex-start' }}>
                  <button type="button" className="btn search"
                    onClick={() => console.log('updated data', this.state.monthData)}
                  > SET</button>
                </div>

              </div>
            </div>
          </div>
        </div>

      </div>

    );
  }
}

export default RPSSettings;